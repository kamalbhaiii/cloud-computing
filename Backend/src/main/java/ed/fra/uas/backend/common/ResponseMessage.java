package ed.fra.uas.backend.common;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ResponseMessage {
    private String message;
    private Object data;
    private LocalDateTime timestamp = LocalDateTime.now();

    public ResponseMessage(String message, Object data) {
        this.message = message;
        this.data = data;
    }
}
