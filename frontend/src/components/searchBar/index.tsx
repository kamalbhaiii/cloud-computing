interface Props {
    onSearch: (term: string) => void;
  }
  
  export default function SearchBar({ onSearch }: Props) {
    return (
      <input
        type="text"
        placeholder="Search by ID, name, metadata..."
        onChange={(e) => onSearch(e.target.value)}
        className="w-full p-2 border rounded-md dark:bg-gray-800 dark:text-white"
      />
    );
  }
  