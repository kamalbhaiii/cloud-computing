import type {ImageItem} from "@/store/imageStore"

const baseImages = [
    {
        image: "https://images.pexels.com/photos/145939/pexels-photo-145939.jpeg?auto=compress&cs=tinysrgb&w=800",
        metadata: "Leopard near river",
    },
    {
        image: "https://images.pexels.com/photos/247431/pexels-photo-247431.jpeg?auto=compress&cs=tinysrgb&w=800",
        metadata: "Elephant family in grassland",
    },
    {
        image: "https://images.pexels.com/photos/301920/pexels-photo-301920.jpeg?auto=compress&cs=tinysrgb&w=800",
        metadata: "Owl staring at camera",
    },
    {
        image: "https://images.pexels.com/photos/53360/tiger-animal-predator-tiger-skin-53360.jpeg?auto=compress&cs=tinysrgb&w=800",
        metadata: "Tiger in the wild",
    },
];

export function generateDummyImages(count: number): ImageItem[] {
    const data: ImageItem[] = [];

    for (let i = 0; i < count; i++) {
        const base = baseImages[i % baseImages.length];
        const now = new Date(Date.now() - i * 10000000);
        const date = now.toLocaleDateString("en-GB");
        const time = now.toLocaleTimeString("en-GB", { hour12: false });
        const name = `image_${i + 1}`;
        data.push({
            id: i + 1,
            image: base.image,
            metadata: base.metadata + ` #${i + 1}`,
            date,
            time,
            name,
            link: base.image,
        });
    }

    return data;
}
