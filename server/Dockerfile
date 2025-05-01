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

# Set the entry point command for the Java application
ENTRYPOINT ["java", "-Dspring.profiles.active=${SPRING_PROFILE}", "-jar", "Backend-1.0.0.jar"]
