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
				sh 'sudo docker-compose build'
			}
		}
		stage("deploy") {
			steps {
				sh "sudo docker-compose up -d"
			}
		}
		stage("Update model") {
			steps {
				sh "sudo docker exec -i msg_guard-web-1 python train.py"
			}
		}
	}
}