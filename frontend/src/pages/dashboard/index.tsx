// src/pages/Dashboard.tsx
import { useEffect, useState } from "react";
import ImageCard from "../../components/card";
import Pagination from "../../components/pagination";
import MetadataModal from "../../components/metadata";
import Alert from "../../components/alert";
import ConfirmModal from "../../components/confirmModal";
import { useImageStore } from "../../store/imageStore";
import { fetchBackendImages, deleteBackendImage, updateBackendImage, deleteBackendMultipleImage } from "../../data/realData";

export default function Dashboard() {
  const { images, setImages, deleteImage, updateImage } = useImageStore();

  const [currentPage, setCurrentPage] = useState(1);
  const [editingImage, setEditingImage] = useState<null | number>(null);
  const [selectedImages, setSelectedImages] = useState<number[]>([]);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [alert, setAlert] = useState({
    show: false,
    message: "",
    type: "success" as "success" | "error",
  });

  const showAlert = (message: string, type: "success" | "error") => {
    setAlert({ show: true, message, type });
    setTimeout(() => setAlert((a) => ({ ...a, show: false })), 3000);
  };

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

  const toggleSelect = (id: number) => {
    setSelectedImages((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

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

  const handleMultipleDelete = async () => {
    const toDelete = images.filter((img) => selectedImages.includes(img.id));
    const names = toDelete.map((img) => img.name);

    try {
      await deleteBackendMultipleImage(names)
      toDelete.forEach((img) => deleteImage(img.id));
      showAlert("Images deleted successfully", "success");
    }
    catch{
      showAlert("Failed to delete selected images", "error");
    } finally {
      setSelectedImages([]);
      setConfirmOpen(false);
    }
  };

  const handleMetadataSave = async (id: number, newMeta: string) => {
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

  const startIndex = (currentPage - 1) * 12;
  const currentImages = images.slice(startIndex, startIndex + 12);

  return (
    <div className="p-6 max-w-7xl mx-auto relative">
      <div className="mb-5 flex justify-between items-center">
  {selectedImages.length > 0 ? (
    <span className="text-gray-600 dark:text-gray-300">
      {selectedImages.length} selected
    </span>
  ) : (
    <span />
  )}

  <div className="flex gap-3">
    {images.length > 0 && (
      <button
        onClick={() => {
          const currentIds = currentImages.map((img) => img.id);
          const allSelected = currentIds.every((id) =>
            selectedImages.includes(id)
          );

          if (allSelected) {
            // Unselect all visible
            setSelectedImages((prev) => prev.filter((id) => !currentIds.includes(id)));
          } else {
            // Select all visible
            setSelectedImages((prev) => [
              ...prev,
              ...currentIds.filter((id) => !prev.includes(id)),
            ]);
          }
        }}
        className="bg-blue-500 hover:bg-blue-700 text-white px-4 py-2 rounded shadow"
      >
        {currentImages.every((img) => selectedImages.includes(img.id))
          ? "Unselect All"
          : "Select All"}
      </button>
    )}

    {selectedImages.length > 0 && (
      <button
        onClick={() => setConfirmOpen(true)}
        className="bg-red-500 hover:bg-red-700 text-white px-4 py-2 rounded shadow"
      >
        Delete Selected ({selectedImages.length})
      </button>
    )}
  </div>
</div>


      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {currentImages.map((img) => (
          <ImageCard
            key={img.id}
            name={img.name}
            image={img.image}
            metadata={img.metadata}
            selected={selectedImages.includes(img.id)}
            onSelectToggle={() => toggleSelect(img.id)}
            onDelete={() => handleDelete(img.id)}
            onEdit={() => setEditingImage(img.id)}
          />
        ))}
      </div>

      <Pagination total={images.length} currentPage={currentPage} onPageChange={setCurrentPage} />

      {editingImage !== null && (
        <MetadataModal
          currentMeta={images.find((i) => i.id === editingImage)?.metadata || ""}
          onClose={() => setEditingImage(null)}
          onSave={(newMeta) => {
            handleMetadataSave(editingImage, newMeta);
            setEditingImage(null);
          }}
        />
      )}

      <Alert message={alert.message} type={alert.type} visible={alert.show} />

      {confirmOpen && (
        <ConfirmModal
          message={`Are you sure you want to delete ${selectedImages.length} image(s)?`}
          onCancel={() => setConfirmOpen(false)}
          onConfirm={handleMultipleDelete}
        />
      )}
    </div>
  );
}
