package ed.fra.uas.backend.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import ed.fra.uas.backend.common.ResponseMessage;
import ed.fra.uas.backend.model.MediaFile;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

@ActiveProfiles("test")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class MediaFileControllerTest {
    @Autowired
    private RestTemplate restTemplate;
    @LocalServerPort
    private int port;
    private static UUID id;
    private static MediaFile mediaFile;
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @BeforeAll
    static void setUp() {
        OBJECT_MAPPER.registerModule(new JavaTimeModule());
        mediaFile = new MediaFile(id, "test MediaFile", "jpg", "/path/to/media/file", "bird, dog, cat", LocalDateTime.now(), LocalDateTime.now());
    }

    @Test
    @Order(1)
    void create() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<MediaFile> request = new HttpEntity<>(mediaFile, headers);
        ResponseEntity<ResponseMessage> response = restTemplate.postForEntity("http://localhost:" + port + "/api/media-files", request, ResponseMessage.class);
        assertEquals(HttpStatus.OK, response.getStatusCode(), "Status code should be 200 OK");
        assertNotNull(response.getBody(), "Response body should not be null");
        assertEquals("Media file created successfully", response.getBody().getMessage(), "The message should be 'Media file created successfully'");
        assertNotNull(response.getBody().getData(), "The data should not be null");
        MediaFile responseMediaFile = OBJECT_MAPPER.convertValue(response.getBody().getData(), MediaFile.class);
        id = responseMediaFile.getId();
    }

    @Test
    @Order(2)
    void getById() {
        ResponseEntity<ResponseMessage> response = restTemplate.getForEntity("http://localhost:" + port + "/api/media-files/" + id, ResponseMessage.class);
        assertEquals(HttpStatus.OK, response.getStatusCode(), "Status code should be 200 OK");
        assertNotNull(response.getBody(), "Response body should not be null");
        assertEquals("Media file retrieved successfully", response.getBody().getMessage(), "The message should be 'Media file retrieved successfully'");
        assertNotNull(response.getBody().getData(), "The data should not be null");
        MediaFile responseMediaFile = OBJECT_MAPPER.convertValue(response.getBody().getData(), MediaFile.class);
        assertEquals(mediaFile.getName(), responseMediaFile.getName(), "The name should match");
        assertEquals(mediaFile.getType(), responseMediaFile.getType(), "The type should match");
        assertEquals(mediaFile.getPath(), responseMediaFile.getPath(), "The path should match");
        assertEquals(mediaFile.getTags(), responseMediaFile.getTags(), "The tags should match");
    }

    @Test
    @Order(3)
    void getAll() {
        ResponseEntity<ResponseMessage> response = restTemplate.getForEntity("http://localhost:" + port + "/api/media-files", ResponseMessage.class);
        assertEquals(HttpStatus.OK, response.getStatusCode(), "Status code should be 200 OK");
        assertNotNull(response.getBody(), "Response body should not be null");
        assertEquals("Media files retrieved successfully", response.getBody().getMessage(), "The message should be 'Media files retrieved successfully'");
        assertNotNull(response.getBody().getData(), "The data should not be null");
    }

    @Test
    @Order(4)
    void update() {
        mediaFile.setName("updated MediaFile");
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<MediaFile> request = new HttpEntity<>(mediaFile, headers);
        ResponseEntity<ResponseMessage> response = restTemplate.exchange("http://localhost:" + port + "/api/media-files", HttpMethod.PUT, request, ResponseMessage.class);
        assertEquals(HttpStatus.OK, response.getStatusCode(), "Status code should be 200 OK");
        assertNotNull(response.getBody(), "Response body should not be null");
        assertEquals("Media file updated successfully", response.getBody().getMessage(), "The message should be 'Media file updated successfully'");
        assertNotNull(response.getBody().getData(), "The data should not be null");
        MediaFile responseMediaFile = OBJECT_MAPPER.convertValue(response.getBody().getData(), MediaFile.class);
        assertEquals(mediaFile.getName(), responseMediaFile.getName(), "The name should match");
    }

@Test
@Order(5)
void delete() {
    HttpHeaders headers = new HttpHeaders();
    headers.setContentType(MediaType.APPLICATION_JSON);
    HttpEntity<Void> request = new HttpEntity<>(headers);

    ResponseEntity<ResponseMessage> response = restTemplate.exchange(
        "http://localhost:" + port + "/api/media-files/" + id,
        HttpMethod.DELETE,
        request,
        ResponseMessage.class
    );

    assertEquals(HttpStatus.OK, response.getStatusCode(), "Status code should be 200 OK");
    assertNotNull(response.getBody(), "Response body should not be null");
    assertEquals("Media file deleted successfully", response.getBody().getMessage(), "The message should be 'Media file deleted successfully'");
    assertNull(response.getBody().getData(), "The data should be null");
}
}