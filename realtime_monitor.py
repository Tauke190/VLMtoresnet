# Real-time Training Monitor
# Uses shared memory for instant updates 
# Automatically starts when training begins

import multiprocessing
import matplotlib.pyplot as plt
import time
import os
import threading
import signal
import sys
from collections import defaultdict
from datetime import datetime

class RealtimeMonitor:
    def __init__(self, plot_path="validation_live.png", max_hours=24, plateau_epochs=5):
        self.plot_path = plot_path
        self.max_hours = max_hours
        self.plateau_epochs = plateau_epochs
        self.start_time = time.time()
        self.best_acc = -1
        self.last_improve_epoch = 0
        self.is_running = False
        
        # Shared memory for training data
        self.manager = multiprocessing.Manager()
        self.training_data = self.manager.dict()
        self.training_data.update({
            'epoch': [],
            'train_loss': [],
            'train_top1': [],
            'train_top5': [],
            'eval_loss': [],
            'eval_top1': [],
            'eval_top5': [],
            'timestamp': [],
            'is_training': False
        })
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def update_metrics(self, epoch, train_metrics, eval_metrics):
        """Update training metrics in shared memory"""
        with self.lock:
            # Manager dict doesn't detect in-place list mutations.
            # Must reassign the whole list for the proxy to sync.
            epochs = list(self.training_data['epoch']) + [epoch]
            self.training_data['epoch'] = epochs
            self.training_data['train_loss'] = list(self.training_data['train_loss']) + [train_metrics.get('loss', 0)]
            self.training_data['train_top1'] = list(self.training_data['train_top1']) + [train_metrics.get('top1', 0)]
            self.training_data['train_top5'] = list(self.training_data['train_top5']) + [train_metrics.get('top5', 0)]
            self.training_data['eval_loss'] = list(self.training_data['eval_loss']) + [eval_metrics.get('loss', 0)]
            self.training_data['eval_top1'] = list(self.training_data['eval_top1']) + [eval_metrics.get('top1', 0)]
            self.training_data['eval_top5'] = list(self.training_data['eval_top5']) + [eval_metrics.get('top5', 0)]
            self.training_data['timestamp'] = list(self.training_data['timestamp']) + [time.time()]
            
    def start_training(self):
        """Signal that training has started"""
        with self.lock:
            self.training_data['is_training'] = True
            self.is_running = True
            
    def stop_training(self):
        """Signal that training has stopped"""
        with self.lock:
            self.training_data['is_training'] = False
            self.is_running = False
            
    def get_current_data(self):
        """Get current training data safely"""
        with self.lock:
            return dict(self.training_data)
            
    def monitor_loop(self):
        """Main monitoring loop - runs in separate thread"""
        print("[START] Real-time monitor started")
        
        while True:
            data = self.get_current_data()
            
            # Check if training is active
            if not data['is_training']:
                time.sleep(10)  # Check every 10 seconds when not training
                continue
                
            # Check if we have data to plot
            if len(data['epoch']) == 0:
                time.sleep(5)
                continue
                
            self.update_plot(data)
            time.sleep(30)  # Update every 30 seconds for real-time monitoring
            
    def update_plot(self, data):
        """Update the monitoring plot with current data"""
        epochs = data['epoch']
        eval_acc = data['eval_top1']
        
        if len(epochs) == 0 or len(eval_acc) == 0:
            return
            
        current_best = max(eval_acc)
        best_epoch = epochs[eval_acc.index(current_best)]
        
        # Detect improvement
        if current_best > self.best_acc:
            self.best_acc = current_best
            self.last_improve_epoch = best_epoch
            
        plateau = (best_epoch - self.last_improve_epoch) >= self.plateau_epochs
        hours = (time.time() - self.start_time) / 3600
        overtime = hours > self.max_hours
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Determine color based on status
        color = "green"
        if plateau:
            color = "orange"
        if overtime:
            color = "red"
            
        # Plot validation accuracy
        plt.subplot(2, 1, 1)
        plt.plot(epochs, eval_acc, marker="o", color=color, label='Validation Top-1')
        plt.axhline(self.best_acc, linestyle="--", alpha=0.7, label=f'Best: {self.best_acc:.2f}%')
        
        # Plot training accuracy if available
        train_acc = data['train_top1']
        if train_acc and any(t > 0 for t in train_acc):
            plt.plot(epochs, train_acc, marker="x", color="blue", alpha=0.7, label='Training Top-1')
            
        plt.title(f"Accuracy Progress - Best={self.best_acc:.2f}% @ epoch {int(best_epoch)}")
        plt.xlabel("Epoch")
        plt.ylabel("Top-1 Accuracy (%)")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Add status indicators
        title_extra = ""
        if plateau:
            title_extra += " [WARNING] Plateau detected"
        if overtime:
            title_extra += " [ALERT] >24hrs"
        if title_extra:
            plt.title(f"Accuracy Progress - Best={self.best_acc:.2f}% @ epoch {int(best_epoch)}{title_extra}")
            
        # Plot loss
        plt.subplot(2, 1, 2)
        eval_loss = data['eval_loss']
        train_loss = data['train_loss']
        
        if eval_loss and any(l > 0 for l in eval_loss):
            plt.plot(epochs, eval_loss, marker="o", color="red", alpha=0.7, label='Validation Loss')
        if train_loss and any(l > 0 for l in train_loss):
            plt.plot(epochs, train_loss, marker="x", color="blue", alpha=0.7, label='Training Loss')
            
        plt.title("Loss Progress")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=140)
        plt.close()
        
        # Critical warnings
        if overtime and plateau:
            print("\n[!!] TRAINING LIKELY WASTING GPU TIME [!!]")
            print("Consider killing this job.\n")
            
    def start_monitor_thread(self):
        """Start monitoring in a separate thread"""
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        return self.monitor_thread

# Global monitor instance
_global_monitor = None

def get_monitor():
    """Get or create global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RealtimeMonitor()
    return _global_monitor

def start_monitoring():
    """Start real-time monitoring - call this when training begins"""
    monitor = get_monitor()
    monitor.start_training()
    monitor.start_monitor_thread()
    print("[INFO] Real-time monitoring activated")
    return monitor

def update_training_metrics(epoch, train_metrics, eval_metrics):
    """Update training metrics - call this after each epoch"""
    monitor = get_monitor()
    monitor.update_metrics(epoch, train_metrics, eval_metrics)

def stop_monitoring():
    """Stop monitoring - call this when training ends"""
    monitor = get_monitor()
    monitor.stop_training()
    print("[INFO] Real-time monitoring stopped")

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print('\n[STOP] Shutting down monitor...')
    stop_monitoring()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    # Standalone mode - for testing
    print("Testing real-time monitor...")
    monitor = start_monitoring()
    
    # Simulate training updates
    for epoch in range(1, 11):
        train_metrics = {'loss': 2.0 - epoch*0.1, 'top1': 20 + epoch*5, 'top5': 40 + epoch*5}
        eval_metrics = {'loss': 1.8 - epoch*0.08, 'top1': 25 + epoch*6, 'top5': 45 + epoch*4}
        update_training_metrics(epoch, train_metrics, eval_metrics)
        time.sleep(2)
        
    stop_monitoring()
    print("Test completed")
