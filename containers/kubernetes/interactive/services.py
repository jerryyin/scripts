#!/usr/bin/env python3
"""Resolve Kubernetes images and setup profiles without Docker or Compose."""
import argparse
import json
import os
from pathlib import Path
import re
import sys


def load_services(config_path):
    config = json.loads(Path(config_path).read_text())
    services = config.get('services', {})
    if not isinstance(services, dict) or not services:
        raise ValueError('config.json must define a nonempty services object')
    resolved = {}
    for name, service in services.items():
        if not isinstance(service, dict):
            raise ValueError(f'Invalid service: {name}')
        profile = service.get('setup_profile', '')
        if not all(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value)
                   for value in (name, profile)):
            raise ValueError(f'Invalid service name or setup_profile: {name}')
        image = os.environ.get(service.get('image_env', '')) or service.get('image', '')
        if not isinstance(image, str) or not re.fullmatch(r'[A-Za-z0-9_./:@-]+', image):
            raise ValueError(f'Invalid image for {name}: {image!r}')
        resolved[name] = (image, profile)
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('service', nargs='?', help='omit to list all services')
    args = parser.parse_args()
    try:
        services = load_services(args.config)
        if args.service:
            if args.service not in services:
                raise ValueError(f'Unknown service {args.service!r}; available: {", ".join(services)}')
            services = {args.service: services[args.service]}
        for name, (image, profile) in services.items():
            print(name, image, profile, sep='\t')
    except (OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
