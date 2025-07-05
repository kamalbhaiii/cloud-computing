import type {ImageItem } from "@/store/imageStore"
import { Pencil, Trash } from "lucide-react";

interface Props {
  item: ImageItem;
  metadata: string;
  onDelete: () => void;
  onEdit: () => void;
}

export default function TableRow({ item, onDelete, onEdit }: Props) {
  return (
    <tr className="hover:bg-gray-100 dark:hover:bg-gray-800">
      <td className="p-2 text-center">{item.id}</td>
      <td className="p-2 text-center">{item.name}</td>
      <td className="p-2 text-center">{item.metadata}</td>
      <td className="p-2 text-center">{item.time}</td>
      <td className="p-2 text-center">
        <a href={item.link} target="_blank" rel="noopener noreferrer" className="text-blue-500 underline">
          View
        </a>
      </td>
      <td className="px-2 py-1">
  <div className="flex justify-evenly items-center space-x-4">
    <button className="text-blue-500 hover:text-blue-700" onClick={onEdit}>
      <Pencil size={16} />
    </button>
    <button className="text-red-500 hover:text-red-700" onClick={onDelete}>
      <Trash size={16} />
    </button>
  </div>
</td>

    </tr>
  );
}
