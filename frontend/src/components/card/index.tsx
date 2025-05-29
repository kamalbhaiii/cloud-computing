import { motion } from "framer-motion";
import { Pencil, Trash } from "lucide-react";

interface Props {
  image: string;
  metadata: string;
  onDelete: () => void;
  onEdit: () => void;
}

export default function ImageCard({ image, metadata, onDelete, onEdit }: Props) {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      className="bg-white dark:bg-gray-800 shadow-md rounded-xl overflow-hidden"
    >
      <img src={image} alt="wildlife" className="w-full h-40 object-cover" />
      <div className="p-4">
        <p className="text-sm text-gray-600 dark:text-gray-300">{metadata}</p>
        <div className="mt-2 flex justify-end space-x-2">
          <button onClick={onEdit} className="text-blue-500 hover:text-blue-700">
            <Pencil size={16} />
          </button>
          <button onClick={onDelete} className="text-red-500 hover:text-red-700">
            <Trash size={16} />
          </button>
        </div>
      </div>
    </motion.div>
  );
}
