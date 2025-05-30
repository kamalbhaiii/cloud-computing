import { create } from "zustand";

type ImageItem = {
    id: number;
    image: string;
    metadata: string;
    date: string;
    time: string;
    name: string; // unique name for matching
    link: string;
}

interface ImageState {
    images: ImageItem[];
    setImages: (images: ImageItem[]) => void;
    updateImage: (id: number, newData: Partial<ImageItem>) => void;
    deleteImage: (id: number) => void;
}

export const useImageStore = create<ImageState>((set) => ({
    images: [],
    setImages: (images) => set({ images }),
    updateImage: (id, newData) =>
        set((state) => ({
            images: state.images.map((img) =>
                img.id === id ? { ...img, ...newData } : img
            ),
        })),
    deleteImage: (id) =>
        set((state) => ({
            images: state.images.filter((img) => img.id !== id),
        })),
}));

export type {ImageItem}