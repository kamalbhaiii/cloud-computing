import { useEffect, useState } from "react";
import { useImageStore } from "../../store/imageStore";
import { fetchBackendImages, deleteBackendImage, updateBackendImage } from "../../data/realData";
import SearchBar from "../../components/searchBar/index";
import TableRow from "../../components/tableRow/index";
import MetadataModal from "../../components/metadata";
import Pagination from "../../components/pagination";
import Alert from "../../components/alert";

export default function Database() {
  const { images, setImages, deleteImage, updateImage } = useImageStore();

  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState<keyof typeof images[0]>("id");
  const [sortAsc, setSortAsc] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [editingImageId, setEditingImageId] = useState<null | number>(null);
  const [alert, setAlert] = useState({
    show: false,
    message: "",
    type: "success" as "success" | "error",
  });

  const itemsPerPage = 12;

  const showAlert = (message: string, type: "success" | "error") => {
    setAlert({ show: true, message, type });
    setTimeout(() => setAlert((a) => ({ ...a, show: false })), 3000);
  };

  useEffect(() => {
    if (images.length === 0) {
      fetchBackendImages()
        .then((data) => setImages(data))
        .catch(() => showAlert("Failed to fetch images from backend", "error"));
    }
  }, [images.length, setImages]);

  const filteredImages = images
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
      }
      return sortAsc
        ? String(aVal).localeCompare(String(bVal))
        : String(bVal).localeCompare(String(aVal));
    });

  const pageStart = (currentPage - 1) * itemsPerPage;
  const paginated = filteredImages.slice(pageStart, pageStart + itemsPerPage);

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

  const handleEdit = async (id: number, newMeta: string) => {
    const image = images.find((img) => img.id === id);
    if (!image) return;

    try {
      await updateBackendImage(image.name, newMeta);
      updateImage(id, { metadata: newMeta });
      showAlert("Image category updated successfully", "success");
    } catch (err) {
      console.error(err);
      showAlert("Failed to update image", "error");
    }
  };

  return (
    <div className="p-4 max-w-7xl mx-auto">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-white">
        Database
      </h2>

      <SearchBar
        onSearch={(term) => {
          setSearchTerm(term);
          setCurrentPage(1); // reset to page 1 on search
        }}
      />

      <div className="overflow-x-auto mt-4 rounded shadow border dark:border-gray-700">
        <table className="min-w-full text-sm text-left dark:text-white">
          <thead className="bg-gray-100 dark:bg-gray-800">
            <tr>
              {["id", "name", "category", "timestamp", "url"].map((field) => (
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
            {paginated.length > 0 ? (
              paginated.map((item) => (
                <TableRow
                  key={item.id}
                  item={item}
                  metadata={item.metadata}
                  onDelete={() => handleDelete(item.id)}
                  onEdit={() => setEditingImageId(item.id)}
                />
              ))
            ) : (
              <tr>
                <td colSpan={6} className="p-4 text-center text-gray-500">
                  No records found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <Pagination
        total={filteredImages.length}
        currentPage={currentPage}
        onPageChange={(page) => setCurrentPage(page)}
      />

      {/* Metadata Modal */}
      {editingImageId !== null && (
        <MetadataModal
          currentMeta={images.find((i) => i.id === editingImageId)?.metadata || ""}
          onClose={() => setEditingImageId(null)}
          onSave={(newMeta) => {
            handleEdit(editingImageId, newMeta);
            setEditingImageId(null);
          }}
        />
      )}

      {/* Alert */}
      <Alert type={alert.type} message={alert.message} visible={alert.show} />
    </div>
  );
}
