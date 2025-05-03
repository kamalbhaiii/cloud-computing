package ed.fra.uas.backend.common;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.web.client.RestTemplate;

@Slf4j
@Order(Ordered.HIGHEST_PRECEDENCE)
@Configuration
public class RestTemplateConfig {
    @Bean
    public RestTemplate restTemplate() {
        log.info("Creating RestTemplate bean");
        return new RestTemplate();
    }
}
