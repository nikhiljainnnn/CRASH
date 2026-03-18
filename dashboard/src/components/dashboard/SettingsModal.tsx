import React, { useState } from 'react';
import { Settings2, X, Save, Smartphone, ShieldCheck } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const [phoneNumber, setPhoneNumber] = useState('+1 (555) 012-3456');
  const [alertThreshold, setAlertThreshold] = useState('0.7');
  const [isSaving, setIsSaving] = useState(false);

  if (!isOpen) return null;

  const handleSave = () => {
    setIsSaving(true);
    // Mock save delay
    setTimeout(() => {
      setIsSaving(false);
      onClose();
    }, 1000);
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in">
      <div className="bg-panel border border-border rounded-xl w-full max-w-md shadow-2xl overflow-hidden shadow-glow-primary">
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border bg-white/[0.02]">
          <h2 className="text-lg font-bold tracking-wide flex items-center gap-2">
            <Settings2 className="text-primary" size={20} />
            System Settings
          </h2>
          <button 
            onClick={onClose}
            className="text-muted hover:text-white transition-colors p-1"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          
          {/* SMS Alerts Section */}
          <div className="space-y-4">
            <h3 className="text-sm font-mono text-muted uppercase tracking-wider flex items-center gap-2">
              <Smartphone size={16} />
              Real-time SMS Alerts
            </h3>
            
            <div className="space-y-2">
              <label className="text-sm text-text/80">Designated Response Number</label>
              <input 
                type="text" 
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                className="w-full bg-black/50 border border-border rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all font-mono"
                placeholder="+1 (555) ..."
              />
              <p className="text-xs text-muted">Immediate SMS notifications will be dispatched to this number when the threshold is exceeded.</p>
            </div>
          </div>

          <div className="h-px bg-border w-full" />

          {/* AI Thresholds */}
          <div className="space-y-4">
            <h3 className="text-sm font-mono text-muted uppercase tracking-wider flex items-center gap-2">
              <ShieldCheck size={16} />
              AI Inference Thresholds
            </h3>
            
            <div className="space-y-2">
              <label className="text-sm text-text/80 flex justify-between">
                <span>Critical Risk Threshold Level</span>
                <span className="text-primary font-mono">{parseFloat(alertThreshold) * 100}%</span>
              </label>
              <input 
                type="range" 
                min="0.5" 
                max="0.95" 
                step="0.05"
                value={alertThreshold}
                onChange={(e) => setAlertThreshold(e.target.value)}
                className="w-full accent-primary"
              />
              <p className="text-xs text-muted">Defines the probability score required to trigger a CRITICAL system event and dispatch SMS/Visual alerts.</p>
            </div>
          </div>

        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border bg-white/[0.02] flex justify-end gap-3">
          <button 
            onClick={onClose}
            className="btn-outline text-sm"
          >
            Cancel
          </button>
          <button 
            onClick={handleSave}
            disabled={isSaving}
            className="btn-primary text-sm flex items-center gap-2"
          >
            {isSaving ? (
              <span className="animate-pulse">Saving...</span>
            ) : (
              <>
                <Save size={16} />
                Save Configuration
              </>
            )}
          </button>
        </div>

      </div>
    </div>
  );
};
