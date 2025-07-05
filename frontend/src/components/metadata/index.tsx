// src/components/MetadataModal.tsx
import { useState } from "react";
import { motion } from "framer-motion";
import Dropdown from "../dropdown";
import { metaOptions } from "./data/metaOptions";

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
        <h2 className="text-xl mb-4 font-semibold">Update Category</h2>
        <Dropdown
          options={metaOptions}
          selected={meta}
          onChange={setMeta}
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
