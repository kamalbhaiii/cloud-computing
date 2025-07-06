// src/components/card.tsx
import { motion } from "framer-motion";
import { Pencil, Trash } from "lucide-react";
import { Chip } from "../chip";

interface Props {
  image: string;
  metadata: string;
  name: string;
  selected?: boolean;
  onSelectToggle?: () => void;
  onDelete: () => void;
  onEdit: () => void;
}

export default function ImageCard({
  image,
  name,
  metadata,
  selected = false,
  onSelectToggle,
  onDelete,
  onEdit,
}: Props) {
  return (
    <motion.div
      whileHover={{ scale: 1.03 }}
      className={`relative bg-white dark:bg-gray-800 shadow-md rounded-xl overflow-hidden border ${
        selected ? "border-blue-500 ring-2 ring-blue-300" : "border-transparent"
      }`}
    >
      {onSelectToggle && (
        <input
          type="checkbox"
          checked={selected}
          onChange={onSelectToggle}
          className="absolute top-2 left-2 z-10 w-5 h-5 accent-blue-500"
        />
      )}

      <img src={image} alt="wildlife" className="w-full h-40 object-cover" />

      <div className="p-4">
        <p className="text-sm text-gray-600 dark:text-gray-300">{name}</p>

        <div className="flex justify-between items-center mt-2">
          <Chip variant="info" label={`Detected: ${metadata}`} />

          <div className="space-x-2">
            <button
              onClick={onEdit}
              className="text-blue-500 hover:text-blue-700"
              aria-label="Edit"
            >
              <Pencil size={16} />
            </button>
            <button
              onClick={onDelete}
              className="text-red-500 hover:text-red-700"
              aria-label="Delete"
            >
              <Trash size={16} />
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
