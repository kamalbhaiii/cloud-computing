import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle, AlertCircle } from "lucide-react";

interface AlertProps {
  type: "success" | "error";
  message: string;
  visible: boolean;
}

export default function Alert({ type, message, visible }: AlertProps) {
  const icon = type === "success" ? <CheckCircle className="text-green-500" /> : <AlertCircle className="text-red-500" />;
  const bg = type === "success" ? "bg-green-100 dark:bg-green-800" : "bg-red-100 dark:bg-red-800";

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -30 }}
          transition={{ duration: 0.4 }}
          className={`fixed top-4 right-4 z-50 shadow-md ${bg} px-4 py-3 rounded-lg flex items-center space-x-2`}
        >
          {icon}
          <span className="text-sm font-medium text-gray-800 dark:text-white">
            {message}
          </span>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
