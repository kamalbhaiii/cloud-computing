import { useState } from "react";
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
  const [isLoading, setIsLoading] = useState(true);

  return (
    <motion.div
      whileHover={{ scale: 1.03 }}
      className={`relative bg-white dark:bg-gray-800 shadow-md rounded-xl overflow-hidden border transition ${
        selected ? "border-blue-500 ring-2 ring-blue-300" : "border-transparent"
      }`}
    >
      {onSelectToggle && (
        <input
          type="checkbox"
          checked={selected}
          onChange={onSelectToggle}
          className="absolute top-2 left-2 z-20 w-5 h-5 accent-blue-500"
        />
      )}

      {/* Full shimmer overlay */}
      {isLoading && (
        <motion.div
          className="absolute inset-0 z-10 bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200 dark:from-gray-700 dark:via-gray-600 dark:to-gray-700 animate-shimmer"
          initial={{ backgroundPosition: "200% 0%" }}
          animate={{ backgroundPosition: ["200% 0%", "-200% 0%"] }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "linear",
          }}
          style={{ backgroundSize: "200% 100%" }}
        />
      )}

      {/* Card content */}
      <div className="w-full h-40 relative">
        <img
          src={image}
          alt="wildlife"
          className="w-full h-40 object-cover"
          onLoad={() => setIsLoading(false)}
        />
      </div>

      <div className="p-4 relative z-0">
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
