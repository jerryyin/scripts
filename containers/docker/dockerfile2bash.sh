#!/bin/bash

# Improved script to convert Dockerfile commands to bash
while read -r line; do
    case $line in
        RUN*)
            echo "${line#RUN }"
            ;;
        ENV*)
            # Convert ENV to export statements
            env_line="${line#ENV }"
            if [[ $env_line == *"="* ]]; then
                # ENV VAR=value format
                echo "export $env_line"
            else
                # ENV VAR value format
                echo "export ${env_line/ /=}"
            fi
            ;;
        WORKDIR*)
            echo "cd ${line#WORKDIR }"
            ;;
        COPY* | ADD*)
            echo "# ${line} (manual intervention required)"
            ;;
        CMD* | ENTRYPOINT*)
            echo "# ${line} (command to run)"
            ;;
        FROM*)
            echo "# ${line} (base image)"
            ;;
        EXPOSE*)
            echo "# ${line} (port exposure, not applicable in bash)"
            ;;
        VOLUME*)
            echo "# ${line} (volume declaration, not directly translatable to bash)"
            ;;
        *)
            echo "# ${line} (unknown instruction)"
            ;;
    esac
done < Dockerfile
