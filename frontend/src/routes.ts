const protocol = window.location.protocol;

export const routes = [
    {
        path: "/",
        name: "Dashboard",
    },
    {
        path: "/database",
        name: "Database",
    },
    ...(protocol === "https:"
        ? [{
            path: "/camera",
            name: "Camera",
        }]
        : []),
    {
        path: "http://grafana.local",
        name: "Monitoring Dashboard",
    },
];
