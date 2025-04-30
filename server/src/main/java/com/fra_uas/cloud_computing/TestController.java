package com.fra_uas.cloud_computing;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TestController {

    @GetMapping("/test")
    public Map<String, String> greet() {
        Map<String, String> response = new HashMap<>();
        response.put("message", "Hey from Spring Boot!!!");
        return response;
    }
}
