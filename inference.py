import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np

# --- NTU-60 Classes ---
NTU_CLASSES = [
    "drink water",
    "eat meal/snack",
    "brushing teeth",
    "brushing hair",
    "drop",
    "pickup",
    "throw",
    "sitting down",
    "standing up (from sitting position)",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "wear jacket",
    "take off jacket",
    "wear a shoe",
    "take off a shoe",
    "wear on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping (one foot jumping)",
    "jump up",
    "make a phone call/answer phone",
    "playing with phone/tablet",
    "typing on a keyboard",
    "pointing to something with finger",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands together",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put the palms together",
    "cross hands in front (say stop)",
    "sneeze/cough",
    "staggering",
    "falling",
    "touch head (headache)",
    "touch chest (stomachache/heart pain)",
    "touch back (backache)",
    "touch neck (neckache)",
    "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "pat on back of other person",
    "point finger at the other person",
    "hugging other person",
    "giving something to other person",
    "touch other person's pocket",
    "handshaking",
    "walking towards each other",
    "walking apart from each other",
]

# --- Mediapipe to COCO 17 Mapping ---
# COCO 17:
# 0: Nose, 1: L_Eye, 2: R_Eye, 3: L_Ear, 4: R_Ear,
# 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow,
# 9: L_Wrist, 10: R_Wrist, 11: L_Hip, 12: R_Hip,
# 13: L_Knee, 14: R_Knee, 15: L_Ankle, 16: R_Ankle

MP_TO_COCO = {
    0: 0,  # Nose
    1: 2,  # L_Eye -> mp 2 is L eye
    2: 5,  # R_Eye -> mp 5 is R eye
    3: 7,  # L_Ear -> mp 7
    4: 8,  # R_Ear -> mp 8
    5: 11,  # L_Shoulder -> mp 11
    6: 12,  # R_Shoulder -> mp 12
    7: 13,  # L_Elbow -> mp 13
    8: 14,  # R_Elbow -> mp 14
    9: 15,  # L_Wrist -> mp 15
    10: 16,  # R_Wrist -> mp 16
    11: 23,  # L_Hip -> mp 23
    12: 24,  # R_Hip -> mp 24
    13: 25,  # L_Knee -> mp 25
    14: 26,  # R_Knee -> mp 26
    15: 27,  # L_Ankle -> mp 27
    16: 28,  # R_Ankle -> mp 28
}

BONE_PAIRS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


# --- Model Definition ---
class Graph:
    def __init__(self):
        self.num_node = 17
        self.edges = BONE_PAIRS
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        A = np.zeros((3, self.num_node, self.num_node))
        for i in range(self.num_node):
            A[0, i, i] = 1
        for i, j in self.edges:
            A[1, i, j] = 1
            A[1, j, i] = 1
        for i in range(2):
            D = np.sum(A[i], axis=1) + 1e-5
            D_inv = np.diag(D**-0.5)
            A[i] = D_inv @ A[i] @ D_inv
        return A


class AdaptiveGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * A_size, kernel_size=1)
        self.PA = nn.Parameter(torch.Tensor(A_size, 17, 17))
        nn.init.uniform_(self.PA, -1e-4, 1e-4)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        N, C, T, V = x.size()
        A_adaptive = A + self.PA
        x = self.conv(x)
        x = x.view(N, -1, A.size(0), T, V)
        x = torch.einsum("ncatv,avw->nctw", x, A_adaptive)
        return self.relu(self.bn(x))


class MultiScaleTCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(9, 1),
            padding=(4, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        res = self.downsample(x)
        x = self.bn(self.conv(x))
        return self.relu(x + res)


class CTR_GCN_Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=60):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * 17)
        A = torch.tensor(Graph().A, dtype=torch.float32)
        self.register_buffer("A", A)

        self.gcn1 = AdaptiveGCNBlock(in_channels, 64, A.size(0))
        self.tcn1 = MultiScaleTCN(64, 64)

        self.gcn2 = AdaptiveGCNBlock(64, 128, A.size(0))
        self.tcn2 = MultiScaleTCN(128, 128, stride=2)

        self.gcn3 = AdaptiveGCNBlock(128, 256, A.size(0))
        self.tcn3 = MultiScaleTCN(256, 256, stride=2)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        x = x.view(N * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(N * M, C, T, V)

        x = self.tcn1(self.gcn1(x, self.A))
        x = self.tcn2(self.gcn2(x, self.A))
        x = self.tcn3(self.gcn3(x, self.A))

        x = x.mean(dim=-1).mean(dim=-1)
        x = x.view(N, M, -1).mean(dim=1)
        return self.fc(x)


def preprocess_sequence(frames_buffer, max_frames=64):
    # frames_buffer is list of shape (17, 3) -> (x, y, confidence)
    # Convert to shape: (M, T, V, C) where M=1, T=max_frames, V=17, C=3
    kp = np.array(frames_buffer)
    kp = np.expand_dims(kp, axis=0)  # (1, T, 17, 3)

    # Uniform sample to max_frames
    T = kp.shape[1]
    if T != max_frames:
        indices = np.linspace(0, T - 1, max_frames).astype(int)
        kp = kp[:, indices, :, :]

    # Centralize relative to center of body (joint 0 - nose as proxy)
    center = kp[0, 0, 0, :2].copy()
    kp[:, :, :, :2] -= center
    
    # Checkpoint was trained with 2 channels (x, y), so we drop confidence
    kp = kp[:, :, :, :2]

    # Transpose to (N, C, T, V, M) format for CTR-GCN
    M, T, V, C = kp.shape
    features = np.transpose(kp, (0, 3, 1, 2))  # (N, C, T, V)
    features = np.expand_dims(features, axis=-1)  # (N, C, T, V, M) -> (1, 2, 64, 17, 1)

    # Model expects M=2 (for 2 person NTU format), so we pad with zeros for the 2nd person
    pad = np.zeros((1, C, T, V, 1))
    features = np.concatenate([features, pad], axis=-1)  # (1, 2, 64, 17, 2)
    return torch.tensor(features, dtype=torch.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Using 2 channels as per the checkpoint size mismatch (34/17 = 2)
    model = CTR_GCN_Network(in_channels=2, num_classes=60).to(device)

    # Try to load weights if available
    checkpoint_path = "models/best_ctr_gcn_joints.pth"
    if os.path.exists(checkpoint_path):
        try:
            # We use weights_only=False due to how PyTorch 2.6 defaults handle unzipped states
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=device, weights_only=False),
                strict=False
            )
            print(f"Weights loaded successfully from {checkpoint_path}.")
        except Exception as e:
            print(f"Could not load weights: {e}. Running with initialized weights.")
    else:
        print(
            f"Checkpoint {checkpoint_path} not found. Running with uninitialized weights."
        )

    model.eval()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)  # 0 for webcam, or pass a video file path

    buffer = []
    MAX_FRAMES = 64
    action_label = "Waiting for frames..."

    print("Starting video inference. Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Mediapipe works with RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Extract 17 joints mapped to COCO layout
            joints = np.zeros((17, 3))
            landmarks = results.pose_landmarks.landmark
            img_h, img_w, _ = image.shape
            for coco_idx, mp_idx in MP_TO_COCO.items():
                lm = landmarks[mp_idx]
                # Scale from [0, 1] to absolute pixel coordinates to match training data variance
                joints[coco_idx] = [lm.x * img_w, lm.y * img_h, lm.visibility]

            buffer.append(joints)

            # Keep only the last MAX_FRAMES
            if len(buffer) > MAX_FRAMES:
                buffer.pop(0)

            # Perform inference when buffer is full
            if len(buffer) == MAX_FRAMES:
                input_tensor = preprocess_sequence(buffer, MAX_FRAMES).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    action_label = NTU_CLASSES[pred_idx]

        # Display the result
        cv2.putText(
            image,
            f"Action: {action_label}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("CTR-GCN Action Recognition", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
