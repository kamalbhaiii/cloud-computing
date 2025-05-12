package ed.fra.uas.backend.service;

import ed.fra.uas.backend.model.MediaFile;
import org.springframework.data.domain.Page;
import org.springframework.stereotype.Service;

import java.util.UUID;

// This test comment-13

/**
 * Service interface for managing MediaFile entities.
 * Provides methods for CRUD operations and pagination.
 */
@Service
public interface MediaFileService {

    /**
     * Retrieves a paginated list of MediaFile entities.
     *
     * @param page the page number to retrieve (0-based index)
     * @param size the number of items per page
     * @return a Page object containing MediaFile entities
     */
    Page<MediaFile> getAll(int page, int size);

    /**
     * Retrieves a MediaFile entity by its unique identifier.
     *
     * @param id the UUID of the MediaFile to retrieve
     * @return the MediaFile entity with the specified ID
     */
    MediaFile getById(UUID id);

    /**
     * Creates a new MediaFile entity.
     *
     * @param mediaFile the MediaFile entity to create
     * @return the created MediaFile entity
     */
    MediaFile create(MediaFile mediaFile);

    /**
     * Updates an existing MediaFile entity.
     *
     * @param mediaFile the MediaFile entity with updated data
     * @return the updated MediaFile entity
     */
    MediaFile update(MediaFile mediaFile);

    /**
     * Deletes a MediaFile entity by its unique identifier.
     *
     * @param id the UUID of the MediaFile to delete
     */
    void delete(UUID id);
}