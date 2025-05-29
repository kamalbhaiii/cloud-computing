pipeline {
    agent any

    environment {
        KUBECTL_CMD = 'sudo kubectl'
    }

    stages {
        stage('Clone Repo') {
            steps {
                git url: 'https://github.com/kamalbhaiii/cloud-computing.git', branch: 'master'
            }
        }

        stage('Deploy K3s Manifests') {
            steps {
                sh '${KUBECTL_CMD} apply -f k3s/'
            }
        }
    }

    post {
        success {
            echo '✅ Deployment successful'
        }
        failure {
            echo '❌ Deployment failed'
        }
    }
}
