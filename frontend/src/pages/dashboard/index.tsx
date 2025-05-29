import { useState } from "react";
import ImageCard from "../../components/card/index";
import Pagination from "../../components/pagination/index";
import MetadataModal from "../../components/metadata/index";
import Alert from "../../components/alert/index";

// Dummy image data
const dummyData = Array.from({ length: 49 }, (_, i) => ({
  id: i,
  image: `https://picsum.photos/seed/${i}/400/300`,
  metadata: `Species #${i + 1}`,
}));

export default function Dashboard() {
  const [images, setImages] = useState(dummyData);
  const [currentPage, setCurrentPage] = useState(1);
  const [editingImage, setEditingImage] = useState<null | number>(null);

  const [alert, setAlert] = useState({ show: false, message: "", type: "success" as "success" | "error" });

  const showAlert = (message: string, type: "success" | "error") => {
    setAlert({ show: true, message, type });
    setTimeout(() => setAlert((a) => ({ ...a, show: false })), 3000);
  };

  const handleDelete = (id: number) => {
    setImages((prev) => prev.filter((img) => img.id !== id));
    showAlert("Image deleted successfully", "success");
  };

  const handleMetadataSave = (id: number, newMeta: string) => {
    setImages((prev) =>
      prev.map((img) => (img.id === id ? { ...img, metadata: newMeta } : img))
    );
    showAlert("Metadata updated", "success");
  };

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
          currentMeta={images.find((i) => i.id === editingImage)?.metadata || ""}
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
