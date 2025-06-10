import { useEffect, useState, useRef, useCallback } from "react";
import { useImageStore } from "../../store/imageStore";
import { fetchBackendImages } from "../../data/realData"; // <-- fetch from backend now
import SearchBar from "../../components/searchBar/index";
import TableRow from "../../components/tableRow/index";
import TableSkeleton from "../../components/tableSkeleton/index";
import { deleteBackendImage } from "../../data/realData";

export default function Database() {
  const { images, setImages, deleteImage, updateImage } = useImageStore();
  const [filtered, setFiltered] = useState(images.slice(0, 20));
  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState<keyof typeof images[0]>("id");
  const [sortAsc, setSortAsc] = useState(true);
  const [hasMore, setHasMore] = useState(true);
  const observer = useRef<IntersectionObserver | null>(null);
  const [alert, setAlert] = useState({
    show: false,
    message: "",
    type: "success" as "success" | "error",
  });

  const showAlert = (message: string, type: "success" | "error") => {
    setAlert({ show: true, message, type });
    setTimeout(() => setAlert((a) => ({ ...a, show: false })), 3000);
  };

  // Fetch images from backend once on mount
  useEffect(() => {
    if (images.length === 0) {
      fetchBackendImages()
        .then((data) => {
          setImages(data);
        })
        .catch(() => {
          showAlert("Failed to fetch images from backend", "error");
        });
    }
  }, [images.length, setImages]);

  // Search + Sort + Slice logic
  useEffect(() => {
    let result = [...images];

    // Search
    if (searchTerm) {
      result = result.filter((img) =>
        Object.values(img).some((val) =>
          val.toString().toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }

    // Sort
    result.sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortAsc ? aVal - bVal : bVal - aVal;
      } else {
        return sortAsc
          ? String(aVal).localeCompare(String(bVal))
          : String(bVal).localeCompare(String(aVal));
      }
    });

    setFiltered(result.slice(0, 20));
    setHasMore(result.length > 20);
  }, [images, searchTerm, sortField, sortAsc]);

  const loadMore = useCallback(() => {
    const nextChunk = 20;
    const newLen = filtered.length + nextChunk;

    const filteredFull = [...images]
      .filter((img) =>
        Object.values(img).some((val) =>
          val.toString().toLowerCase().includes(searchTerm.toLowerCase())
        )
      )
      .sort((a, b) => {
        const aVal = a[sortField];
        const bVal = b[sortField];
        if (typeof aVal === "number" && typeof bVal === "number") {
          return sortAsc ? aVal - bVal : bVal - aVal;
        } else {
          return sortAsc
            ? String(aVal).localeCompare(String(bVal))
            : String(bVal).localeCompare(String(aVal));
        }
      });

    const more = filteredFull.slice(0, newLen);
    setFiltered(more);
    setHasMore(more.length < filteredFull.length);
  }, [filtered.length, images, searchTerm, sortField, sortAsc]);

  const lastRowRef = useCallback(
    (node: HTMLTableRowElement | null) => {
      if (observer.current) observer.current.disconnect();
      observer.current = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && hasMore) {
          loadMore();
        }
      });
      if (node) observer.current.observe(node);
    },
    [hasMore, loadMore]
  );

  // Edit and Delete handlers
  const handleDelete = async (id: number) => {
    const image = images.find((img) => img.id === id);
    if (!image) return;
  
    try {
      await deleteBackendImage(image.name);
      deleteImage(id);
      showAlert("Image deleted successfully", "success");
    } catch (err) {
      console.error(err);
      showAlert("Failed to delete image", "error");
    }
  };
  
  const handleEdit = (id: number, newMeta: string) => {
    updateImage(id, { metadata: newMeta });
    showAlert("Metadata updated", "success");
  };

  return (
    <div className="p-4 max-w-7xl mx-auto">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-white">
        Image Metadata Table
      </h2>
      <SearchBar onSearch={(term) => setSearchTerm(term)} />
      <div className="overflow-x-auto mt-4 rounded shadow border dark:border-gray-700">
        <table className="min-w-full text-sm text-left dark:text-white">
          <thead className="bg-gray-100 dark:bg-gray-800">
            <tr>
              {["id", "date", "time", "name", "link"].map((field) => (
                <th
                  key={field}
                  onClick={() => {
                    setSortField(field as keyof typeof images[0]);
                    setSortAsc((prev) => (sortField === field ? !prev : true));
                  }}
                  className="p-2 cursor-pointer select-none hover:bg-gray-200 dark:hover:bg-gray-700"
                >
                  {field.toUpperCase()}
                  {sortField === field && (
                    <span className="ml-1">{sortAsc ? "↑" : "↓"}</span>
                  )}
                </th>
              ))}
              <th className="p-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((item, idx) =>
              idx === filtered.length - 1 ? (
                <tr ref={lastRowRef} key={item.id}>
                  <TableRow
                    item={item}
                    metadata={item.metadata}
                    onDelete={() => handleDelete(item.id)}
                    onEdit={() => {
                      const newMeta = prompt(
                        "Edit metadata",
                        item.metadata
                      );
                      if (newMeta !== null && newMeta.trim() !== "") {
                        handleEdit(item.id, newMeta.trim());
                      }
                    }}
                  />
                </tr>
              ) : (
                <TableRow
                  key={item.id}
                  item={item}
                  metadata={item.metadata}
                  onDelete={() => handleDelete(item.id)}
                  onEdit={() => {
                    const newMeta = prompt("Edit metadata", item.metadata);
                    if (newMeta !== null && newMeta.trim() !== "") {
                      handleEdit(item.id, newMeta.trim());
                    }
                  }}
                />
              )
            )}
            {hasMore && <TableSkeleton />}
          </tbody>
        </table>
      </div>

      {/* Alert component for showing success/error messages */}
      {alert.show && (
        <div
          className={`fixed bottom-4 right-4 px-4 py-2 rounded shadow ${
            alert.type === "success"
              ? "bg-green-500 text-white"
              : "bg-red-500 text-white"
          }`}
        >
          {alert.message}
        </div>
      )}
    </div>
  );
}
