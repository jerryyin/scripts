cat <<EOF > /etc/docker/daemon.json
{
    "insecure-registries": [
	"compute-artifactory.amd.com:5000",
	"compute-artifactory.amd.com:5001"
    ],
    "storage-driver": "overlay2"
}
EOF

service docker restart

