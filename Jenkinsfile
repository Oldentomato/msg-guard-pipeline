pipeline {
	agent any
	stages {
		stage("Checkout") {
			steps {
				checkout scm
			}
		}
		stage("Build") {
			steps {
				sh 'docker-compose build web'
			}
		}
		stage("deploy") {
			steps {
				sh "docker-compose up -d"
			}
		}
		stage("Update model") {
			steps {
				sh "docker exec -i msg_guard-web-1 python train.py"
			}
		}
	}
}