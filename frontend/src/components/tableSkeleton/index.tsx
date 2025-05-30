export default function TableSkeleton() {
    return (
      <tr>
        <td colSpan={5} className="p-4 text-center text-gray-500 dark:text-gray-400">
          Loading more data...
        </td>
      </tr>
    );
  }
  