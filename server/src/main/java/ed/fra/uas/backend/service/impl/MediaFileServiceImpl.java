package ed.fra.uas.backend.service.impl;

import ed.fra.uas.backend.model.MediaFile;
import ed.fra.uas.backend.repository.MediaFileRepository;
import ed.fra.uas.backend.service.MediaFileService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

import java.util.UUID;

@Slf4j
@Service
public class MediaFileServiceImpl implements MediaFileService {
    private final MediaFileRepository mediaFileRepository;

    protected MediaFileServiceImpl(MediaFileRepository mediaFileRepository) {
        this.mediaFileRepository = mediaFileRepository;
    }

    /**
     * Retrieves a paginated list of MediaFile entities.
     *
     * @param page the page number to retrieve (0-based index)
     * @param size the number of items per page
     * @return a Page object containing MediaFile entities
     */
    public Page<MediaFile> getAll(int page, int size) {
        log.info("Retrieving all media files, page: {}, size: {}", page, size);
        return mediaFileRepository.findAll(PageRequest.of(page, size));
    }

    /**
     * Retrieves a MediaFile entity by its unique identifier.
     *
     * @param id the UUID of the MediaFile to retrieve
     * @return the MediaFile entity with the specified ID
     */
    public MediaFile getById(UUID id) {
        log.info("Retrieving media file with ID: {}", id);
        return mediaFileRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Media file not found"));
    }

    /**
     * Creates a new MediaFile entity.
     *
     * @param mediaFile the MediaFile entity to create
     * @return the created MediaFile entity
     */
    public MediaFile create(MediaFile mediaFile) {
        log.info("Creating new media file: {}", mediaFile);
        return mediaFileRepository.save(mediaFile);
    }

    /**
     * Updates an existing MediaFile entity.
     *
     * @param mediaFile the MediaFile entity with updated data
     * @return the updated MediaFile entity
     */
    public MediaFile update(MediaFile mediaFile) {
        log.info("Updating media file: {}", mediaFile);
        return mediaFileRepository.save(mediaFile);
    }

    /**
     * Deletes a MediaFile entity by its unique identifier.
     *
     * @param id the UUID of the MediaFile to delete
     */
    public void delete(UUID id) {
        log.info("Deleting media file with ID: {}", id);
        if (!mediaFileRepository.existsById(id)) {
            throw new RuntimeException("Media file not found");
        }
        mediaFileRepository.deleteById(id);
    }
}
