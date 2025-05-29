pipeline {
    agent any

    environment {
        KUBECONFIG = '/var/lib/jenkins/.kube/config'
    }

    stages {
        stage('Clone Repo') {
            steps {
                git url: 'https://github.com/kamalbhaiii/cloud-computing.git', branch: 'master'
            }
        }

        stage('Deploy K3s Manifests') {
            steps {
                sh 'sudo /usr/local/bin/kubectl apply -f k3s/'
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
