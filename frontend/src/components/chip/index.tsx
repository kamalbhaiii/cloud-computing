import { X, CheckCircle, AlertTriangle, Info } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils'; // Utility for classNames

type ChipProps = {
  label: string;
  variant?: 'default' | 'success' | 'error' | 'info';
  icon?: React.ReactNode;
  onRemove?: () => void;
  className?: string;
};

const variantStyles = {
  default: 'bg-muted text-foreground',
  success: 'bg-green-100 text-green-800',
  error: 'bg-red-100 text-red-800',
  info: 'bg-blue-100 text-blue-800',
};

const variantIcons: Record<'default' | 'success' | 'error' | 'info', React.ReactNode> = {
    default: null, // or use a neutral icon if you prefer
    success: <CheckCircle size={14} className="mr-1" />,
    error: <AlertTriangle size={14} className="mr-1" />,
    info: <Info size={14} className="mr-1" />,
  };
  

export const Chip: React.FC<ChipProps> = ({
  label,
  variant = 'default',
  icon,
  onRemove,
  className,
}) => {
  const baseIcon = icon ?? variantIcons[variant];

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      exit={{ scale: 0.7, opacity: 0 }}
      transition={{ duration: 0.2 }}
      className={cn(
        'inline-flex items-center px-3 py-1 rounded-full text-sm font-medium shadow-sm',
        variantStyles[variant],
        className
      )}
    >
      {baseIcon && <span className="mr-1">{baseIcon}</span>}
      <span>{label}</span>
      {onRemove && (
        <button
          onClick={onRemove}
          className="ml-2 hover:opacity-80 transition-opacity"
        >
          <X size={14} />
        </button>
      )}
    </motion.div>
  );
};
