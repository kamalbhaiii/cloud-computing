import type { ImageItem } from "@/store/imageStore";
import config from '../config/envConfig'

export async function fetchBackendImages(): Promise<ImageItem[]> {
    try {
        const protocol = window.location.protocol;
        console.log(protocol)
        const response = await fetch(`${protocol == 'https:' ? config.SECURE_BACKEND_END_POINT : config.BACKEND_END_POINT}:${config.BACKEND_PORT}/api/images`);
        const result = await response.json();

        if (result.status !== "success" || !Array.isArray(result.data)) {
            throw new Error("Invalid response format");
        }

        const images: ImageItem[] = result.data.map((item:any) => {
            const timestamp = new Date(item.timestamp);
            const date = timestamp.toLocaleDateString("en-GB");
            const time = timestamp.toLocaleTimeString("en-GB", { hour12: false });

            return {
                id: item.id,
                name: item.name,
                image: item.url,
                metadata: item.category,
                date,
                time,
                link: item.url,
            };
        });

        return images;
    } catch (error) {
        console.error("Error fetching backend images:", error);
        return [];
    }
}


export async function deleteBackendImage(name: string): Promise<void> {
    const res = await fetch(`${config.BACKEND_END_POINT}:${config.BACKEND_PORT ? config.BACKEND_PORT : ''}/api/images/${name}`, {
        method: "DELETE",
    });

    if (!res.ok) {
        throw new Error("Failed to delete image from backend");
    }
}
  