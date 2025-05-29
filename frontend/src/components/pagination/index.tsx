interface Props {
    total: number;
    currentPage: number;
    onPageChange: (page: number) => void;
  }
  
  export default function Pagination({ total, currentPage, onPageChange }: Props) {
    const totalPages = Math.ceil(total / 12);
  
    return (
      <div className="flex justify-center mt-6">
        {Array.from({ length: totalPages }, (_, i) => (
          <button
            key={i}
            onClick={() => onPageChange(i + 1)}
            className={`mx-1 px-3 py-1 rounded ${
              currentPage === i + 1 ? "bg-blue-500 text-white" : "bg-gray-200"
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>
    );
  }
  