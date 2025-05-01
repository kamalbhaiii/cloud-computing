package ed.fra.uas.backend.controller;

import ed.fra.uas.backend.common.ResponseMessage;
import ed.fra.uas.backend.model.MediaFile;
import ed.fra.uas.backend.service.MediaFileService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.UUID;

/**
 * Controller for managing MediaFile entities.
 * Provides endpoints for CRUD operations and retrieval of media files.
 */
@Slf4j
@RestController
@RequestMapping("/api/media-files")
public class MediaFileController {
    private final MediaFileService mediaFileService;

    /**
     * Constructor for MediaFileController.
     *
     * @param mediaFileService the service used to manage MediaFile entities
     */
    public MediaFileController(MediaFileService mediaFileService) {
        this.mediaFileService = mediaFileService;
    }

    /**
     * Retrieves all media files with pagination.
     *
     * @return a ResponseEntity containing a ResponseMessage with the list of media files
     */
    @GetMapping
    public ResponseEntity<ResponseMessage> getAll() {
        log.info("Retrieving all media files");
        return ResponseEntity.ok(new ResponseMessage("Media files retrieved successfully", mediaFileService.getAll(0, 10)));
    }

    /**
     * Retrieves a specific media file by its ID.
     *
     * @param id the UUID of the media file to retrieve
     * @return a ResponseEntity containing a ResponseMessage with the requested media file
     */
    @GetMapping("/{id}")
    public ResponseEntity<ResponseMessage> getById(@PathVariable UUID id) {
        log.info("Retrieving media file with ID: {}", id);
        return ResponseEntity.ok(new ResponseMessage("Media file retrieved successfully", mediaFileService.getById(id)));
    }

    /**
     * Creates a new media file.
     *
     * @param mediaFile the MediaFile object to create
     * @return a ResponseEntity containing a ResponseMessage with the created media file
     */
    @PostMapping()
    public ResponseEntity<ResponseMessage> create(@RequestBody MediaFile mediaFile) {
        log.info("Creating new media file");
        return ResponseEntity.ok(new ResponseMessage("Media file created successfully", mediaFileService.create(mediaFile)));
    }

    /**
     * Updates an existing media file.
     *
     * @param mediaFile the MediaFile object with updated data
     * @return a ResponseEntity containing a ResponseMessage with the updated media file
     */
    @PutMapping
    public ResponseEntity<ResponseMessage> update(@RequestBody MediaFile mediaFile) {
        log.info("Updating media file with ID: {}", mediaFile.getId());
        return ResponseEntity.ok(new ResponseMessage("Media file updated successfully", mediaFileService.update(mediaFile)));
    }

    /**
     * Deletes a media file by its ID.
     *
     * @param id the UUID of the media file to delete
     * @return a ResponseEntity containing a ResponseMessage indicating the deletion status
     */
    @DeleteMapping
    public ResponseEntity<ResponseMessage> delete(@PathVariable UUID id) {
        log.info("Deleting media file with ID: {}", id);
        mediaFileService.delete(id);
        return ResponseEntity.ok(new ResponseMessage("Media file deleted successfully", null));
    }
}