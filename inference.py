"""
Inference and Evaluation for Action Recognition Model
"""

import torch
import json
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from action_recognition_classifier import SkeletonLSTM


class ActionRecognitionInference:
    """Inference pipeline for trained models"""
    
    def __init__(self, model_path, config_path, scaler_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.action_map = {int(k): v for k, v in self.config['action_map'].items()}
        self.action_names = {v: k for k, v in self.action_map.items()}
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Build and load model
        self.model = SkeletonLSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            num_classes=self.config['num_classes']
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Number of classes: {self.config['num_classes']}")
        print(f"Device: {self.device}")
    
    def predict_single(self, skeleton_array):
        """
        Predict action for a single skeleton sequence
        
        Args:
            skeleton_array: (seq_length, num_joints*3) numpy array
        
        Returns:
            action_id, confidence, top_3_predictions
        """
        # Normalize
        skeleton_array = self.scaler.transform(skeleton_array)
        
        # Pad/truncate
        if len(skeleton_array) >= 300:
            skeleton_array = skeleton_array[:300]
        else:
            padding = np.zeros((300 - len(skeleton_array), skeleton_array.shape[1]))
            skeleton_array = np.vstack([skeleton_array, padding])
        
        # Forward
        with torch.no_grad():
            x = torch.FloatTensor(skeleton_array).unsqueeze(0).to(self.device)
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            
            top_3_probs, top_3_indices = torch.topk(probs, 3, dim=1)
            
            predicted_id = top_3_indices[0, 0].item()
            confidence = top_3_probs[0, 0].item()
            
            top_3 = [
                (self.action_names[idx.item()], prob.item())
                for idx, prob in zip(top_3_indices[0], top_3_probs[0])
            ]
        
        return predicted_id, confidence, top_3
    
    def evaluate_dataset(self, test_loader):
        """Evaluate on full test set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels, _ in test_loader:
                sequences = sequences.to(self.device)
                labels = labels.squeeze()
                
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def generate_report(self, test_loader, output_dir='./'):
        """Generate comprehensive evaluation report"""
        print("Evaluating model...")
        predictions, ground_truth = self.evaluate_dataset(test_loader)
        
        accuracy = accuracy_score(ground_truth, predictions)
        
        print(f"\n{'='*60}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"{'='*60}\n")
        
        print("Classification Report:")
        print(classification_report(
            ground_truth, predictions,
            target_names=[str(self.action_names[i]) for i in range(self.config['num_classes'])],
            zero_division=0
        ))
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions, 
                            labels=list(range(self.config['num_classes'])))
        
        # Plot confusion matrix
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=False, cmap='Blues', cbar=True, 
                   xticklabels=range(self.config['num_classes']),
                   yticklabels=range(self.config['num_classes']))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=150)
        print("✓ Confusion matrix saved")
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(14, 6))
        plt.bar(range(self.config['num_classes']), per_class_acc)
        plt.xlabel('Action Class')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/per_class_accuracy.png', dpi=150)
        print("✓ Per-class accuracy plot saved")
        
        return {
            'overall_accuracy': accuracy,
            'per_class_accuracy': per_class_acc.tolist(),
            'predictions': predictions.tolist(),
            'ground_truth': ground_truth.tolist()
        }


def example_usage():
    """Example: How to use the inference pipeline"""
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize inference
    inference = ActionRecognitionInference(
        model_path='final_model.pth',
        config_path='model_config.json',
        scaler_path='scaler.pkl',
        device=device
    )
    
    # Option 1: Predict single skeleton sequence
    print("\n" + "="*60)
    print("Example 1: Single Skeleton Prediction")
    print("="*60)
    
    # Create dummy skeleton data (seq_length=100, 25 joints * 3 coords = 75 features)
    dummy_skeleton = np.random.randn(100, 75)
    
    action_id, confidence, top_3 = inference.predict_single(dummy_skeleton)
    
    print(f"\nPredicted Action ID: {action_id}")
    print(f"Confidence: {confidence:.4f}")
    print("\nTop 3 Predictions:")
    for action_name, prob in top_3:
        print(f"  - Action {action_name}: {prob:.4f}")
    
    # Option 2: Evaluate on test set
    print("\n" + "="*60)
    print("Example 2: Evaluate on Test Set")
    print("="*60)
    
    # Load test data (you'll need to set this up)
    # test_loader = DataLoader(test_dataset, batch_size=32)
    # report = inference.generate_report(test_loader)
    
    print("\nTo evaluate on test set, use:")
    print("  report = inference.generate_report(test_loader)")


if __name__ == "__main__":
    example_usage()
