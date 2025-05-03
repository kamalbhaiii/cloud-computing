# README: Docker Build and Usage with Docker Compose

## Overview
This project uses Docker to deploy the application in a containerized environment. The configuration supports environment variables that can be set during build and runtime to make the application flexible.

---

## Docker Build Process

### Steps to Build the Docker Image:
1. **Understand the `Dockerfile`**:
    - The `Dockerfile` consists of two stages:
        1. **Build Phase**: Uses Maven to download dependencies and build the project.
        2. **Runtime Phase**: Uses a lightweight OpenJDK image to run the application.

2. **Build the Image**:
   Run the following command to build the Docker image:
   ```bash
   docker build -t backend-app .
   ```

3. **Important Arguments in the Dockerfile**:
    - `SPRING_PROFILE`: Specifies the active Spring profile (e.g., `prod`).
    - `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USERNAME`, `DB_PASSWORD`: Database configuration variables.
    - `SERVER_ADDRESS`, `SERVER_PORT`: Server address and port.

---

## Docker Compose

### Using Docker Compose:
Docker Compose allows you to easily start the application and its dependencies (e.g., the database).

1. **Example `docker-compose.yml`**:
   ```yaml
   version: '3.8'

   services:
     backend:
       build:
         context: .
         dockerfile: Dockerfile
       environment:
         SPRING_PROFILE: prod
         DB_HOST: db
         DB_PORT: 3306
         DB_NAME: cloud-ai
         DB_USERNAME: root
         DB_PASSWORD: cloud-ai
         SERVER_ADDRESS: 0.0.0.0
         SERVER_PORT: 9090
       ports:
         - "9090:9090"
       depends_on:
         - db

     db:
       image: mysql:8.0
       environment:
         MYSQL_ROOT_PASSWORD: cloud-ai
         MYSQL_DATABASE: cloud-ai
       ports:
         - "3306:3306"
   ```

2. **Start the Application**:
   Run the following command to start the application with Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. **Stop the Application**:
   ```bash
   docker-compose down
   ```

---

## Environment Variables

| Variable         | Description                          | Example Value      |
|-------------------|--------------------------------------|--------------------|
| `SPRING_PROFILE`  | Active Spring profile               | `prod`             |
| `DB_HOST`         | Database hostname                   | `db`               |
| `DB_PORT`         | Database port                       | `3306`             |
| `DB_NAME`         | Database name                       | `cloud-ai`         |
| `DB_USERNAME`     | Database username                   | `root`             |
| `DB_PASSWORD`     | Database password                   | `cloud-ai`         |
| `SERVER_ADDRESS`  | Server address                      | `0.0.0.0`          |
| `SERVER_PORT`     | Server port                         | `9090`             |

---

## Notes
- Ensure Docker and Docker Compose are installed on your system.
- Adjust the environment variables in the `docker-compose.yml` file to match your environment.
- Use `application-prod.yml` to define the configuration for the `prod` profile.

---

## API Documentation: MediaFileController

The `MediaFileController` provides endpoints for managing `MediaFile` entities. Below is an overview of the available endpoints:

### Endpoints

| HTTP Method | Endpoint               | Description                              |
|-------------|------------------------|------------------------------------------|
| GET         | `/api/media-files`     | Retrieves all media files with pagination. |
| GET         | `/api/media-files/{id}`| Retrieves a specific media file by its ID. |
| POST        | `/api/media-files`     | Creates a new media file.                |
| PUT         | `/api/media-files`     | Updates an existing media file.          |
| DELETE      | `/api/media-files/{id}`| Deletes a media file by its ID.          |

### Notes
- Replace `{id}` with the UUID of the media file when using the endpoints that require an ID.
- The API returns responses wrapped in a `ResponseMessage` object, which includes a message and the requested data.

---

## Entity Documentation: MediaFile

The `MediaFile` entity represents a media file in the system. Below is an overview of its attributes:

### Attributes

| Attribute   | Type            | Description                              |
|-------------|-----------------|------------------------------------------|
| `id`        | `UUID`          | Unique identifier for the media file.    |
| `name`      | `String`        | Name of the media file.                  |
| `type`      | `String`        | Type of the media file (e.g., image, video). |
| `path`      | `String`        | Path to the media file.                  |
| `tags`      | `String`        | Tags associated with the media file.     |
| `createdAt` | `LocalDateTime` | Timestamp when the media file was created. |
| `updatedAt` | `LocalDateTime` | Timestamp when the media file was last updated. |

### Notes
- The `id` field is auto-generated and serves as the primary key.
- The `createdAt` and `updatedAt` fields are automatically managed by the system.
- The `path` field stores the location of the media file.
- The `tags` field can be used to categorize or describe the media file.e `content` field stores the raw binary data of the media file.