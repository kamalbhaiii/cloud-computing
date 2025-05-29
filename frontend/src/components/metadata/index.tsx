import { useState } from "react";
import { motion } from "framer-motion";

interface Props {
  currentMeta: string;
  onClose: () => void;
  onSave: (newMeta: string) => void;
}

export default function MetadataModal({ currentMeta, onClose, onSave }: Props) {
  const [meta, setMeta] = useState(currentMeta);

  return (
    <motion.div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl shadow-lg w-96">
        <h2 className="text-xl mb-4 font-semibold">Edit Metadata</h2>
        <input
          value={meta}
          onChange={(e) => setMeta(e.target.value)}
          className="w-full p-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-800"
        />
        <div className="mt-4 flex justify-end space-x-2">
          <button onClick={onClose} className="px-3 py-1 bg-gray-300 rounded">
            Cancel
          </button>
          <button
            onClick={() => {
              onSave(meta);
              onClose();
            }}
            className="px-3 py-1 bg-blue-500 text-white rounded"
          >
            Save
          </button>
        </div>
      </div>
    </motion.div>
  );
}
