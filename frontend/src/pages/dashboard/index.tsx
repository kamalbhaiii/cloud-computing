import { useEffect, useState } from "react";
import ImageCard from "../../components/card";
import Pagination from "../../components/pagination";
import MetadataModal from "../../components/metadata";
import Alert from "../../components/alert";

import { useImageStore } from "../../store/imageStore";
import { generateDummyImages } from "../../data/dummyData";

export default function Dashboard() {
  const { images, setImages, deleteImage, updateImage } = useImageStore();

  const [currentPage, setCurrentPage] = useState(1);
  const [editingImage, setEditingImage] = useState<null | number>(null);
  const [alert, setAlert] = useState({
    show: false,
    message: "",
    type: "success" as "success" | "error",
  });

  const showAlert = (message: string, type: "success" | "error") => {
    setAlert({ show: true, message, type });
    setTimeout(() => setAlert((a) => ({ ...a, show: false })), 3000);
  };

  // Initialize dummy images only once
  useEffect(() => {
    if (images.length === 0) {
      setImages(generateDummyImages(49));
    }
  }, [images.length, setImages]);

  const handleDelete = (id: number) => {
    deleteImage(id);
    showAlert("Image deleted successfully", "success");
  };

  const handleMetadataSave = (id: number, newMeta: string) => {
    updateImage(id, { metadata: newMeta });
    showAlert("Metadata updated", "success");
  };

  // Pagination
  const startIndex = (currentPage - 1) * 12;
  const currentImages = images.slice(startIndex, startIndex + 12);

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {currentImages.map((img) => (
          <ImageCard
            key={img.id}
            image={img.image}
            metadata={img.metadata}
            onDelete={() => handleDelete(img.id)}
            onEdit={() => setEditingImage(img.id)}
          />
        ))}
      </div>

      <Pagination
        total={images.length}
        currentPage={currentPage}
        onPageChange={setCurrentPage}
      />

      {editingImage !== null && (
        <MetadataModal
          currentMeta={
            images.find((i) => i.id === editingImage)?.metadata || ""
          }
          onClose={() => setEditingImage(null)}
          onSave={(newMeta) => {
            handleMetadataSave(editingImage, newMeta);
            setEditingImage(null);
          }}
        />
      )}

      <Alert message={alert.message} type={alert.type} visible={alert.show} />
    </div>
  );
}
