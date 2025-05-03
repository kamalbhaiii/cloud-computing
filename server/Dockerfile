ARG MVN_VERSION=3.8.5-openjdk-17
ARG JDK_VERSION=17
ARG maven_tool_chain_cache

FROM maven:${MVN_VERSION} AS maven_tool_chain_cache
WORKDIR /build
COPY pom.xml .
COPY src ./src/

RUN mvn dependency:go-offline
RUN mvn clean package --quiet -DskipTests --batch-mode --fail-fast --no-transfer-progress

FROM openjdk:${JDK_VERSION}-jdk

COPY --from=maven_tool_chain_cache /build/target/ /

ENV SPRING_PROFILE=prod
ENV DB_HOST=10.20.30.3
ENV DB_PORT=3306
ENV DB_NAME=cloud-ai
ENV DB_USERNAME=root
ENV DB_PASSWORD=cloud-ai
ENV SERVER_ADDRESS=10.20.30.4
ENV SERVER_PORT=9090

# Set the entry point command for the Java application
ENTRYPOINT ["java", "-Dspring.profiles.active=${SPRING_PROFILE}", "-jar", "Backend-1.0.0.jar"]
