pipeline {
    agent any

    environment {
        GIT_DISCOVERY_ACROSS_FILESYSTEM = '1'
        PROJECT_DIR = '/home/ubuntu/cloud-computing'
    }

    stages {
        stage('Pull Latest Changes') {
            steps {
                echo "Checking out latest changes on master branch..."
                dir("${PROJECT_DIR}") {
                    sh '''
                    whoami
                    ls -la
                    git checkout master
                    git pull origin master
                    '''
                }
            }
        }

        stage('Deploy to k3s') {
            steps {
                echo "Applying k3s manifests..."
                dir("${PROJECT_DIR}/k3s") {
                    sh '''
                    ls -la
                    kubectl apply -f .
                    '''
                }
            }
        }
    }
}