"""
Test data generator for integration tests
Creates realistic test data for pipeline testing
"""

import json
from pathlib import Path
from datetime import datetime
import random
import string


class FakeDataGenerator:
    """Generate test data for pipeline testing"""
    
    def __init__(self):
        self.platforms = [1, 2, 3]
        self.videos = [1, 2, 3]
        self.sessions = [1, 2]
        
    def generate_user_id(self, prefix=""):
        """Generate a valid 32-character hex user ID"""
        # Ensure the entire ID is hex characters only
        hex_id = "".join(random.choices("0123456789abcdef", k=32))
        return hex_id
    
    def generate_keystroke_data(self):
        """Generate realistic keystroke data"""
        lines = []
        timestamp = 100
        
        # Generate some keystrokes
        for _ in range(20):
            key = random.choice(string.ascii_lowercase)
            lines.append(f"P,{key},{timestamp}")
            timestamp += random.randint(50, 150)
            lines.append(f"R,{key},{timestamp}")
            timestamp += random.randint(30, 100)
            
        return "\n".join(lines)
    
    def generate_user_files(self, output_dir: Path, user_id: str, num_files: int = 18):
        """Generate user files (partial or complete)"""
        user_id = self.generate_user_id(user_id[:10] if len(user_id) > 10 else user_id)
        
        # Generate metadata files
        self.generate_metadata_files(output_dir, user_id)
        
        # Map platform numbers to letters
        platform_letters = {1: 'f', 2: 'i', 3: 't'}  # Facebook, Instagram, Twitter
        
        # Generate keystroke files (up to num_files)
        count = 0
        sequence = 0
        for session in self.sessions:
            for video in self.videos:
                for platform in self.platforms:
                    if count >= num_files:
                        return user_id
                    
                    # Use web app format: {platform_letter}_{user_id}_{sequence}.csv
                    platform_letter = platform_letters[platform]
                    filename = f"{platform_letter}_{user_id}_{sequence}.csv"
                    filepath = output_dir / filename
                    
                    with open(filepath, 'w') as f:
                        f.write(self.generate_keystroke_data())
                    
                    count += 1
                    sequence += 1
                    
        return user_id
    
    def generate_metadata_files(self, output_dir: Path, user_id: str):
        """Generate required metadata files"""
        # Consent form (using correct name)
        consent_data = {
            "user_id": user_id,
            "consent": True,
            "timestamp": datetime.now().isoformat()
        }
        with open(output_dir / f"{user_id}_consent.json", 'w') as f:
            json.dump(consent_data, f)
        
        # Demographics
        demographics_data = {
            "user_id": user_id,
            "age": random.randint(18, 65),
            "gender": random.choice(["M", "F", "Other"]),
            "typing_experience": random.choice(["Beginner", "Intermediate", "Expert"])
        }
        with open(output_dir / f"{user_id}_demographics.json", 'w') as f:
            json.dump(demographics_data, f)
        
        # Start time
        start_time_data = {
            "user_id": user_id,
            "start_time": datetime.now().isoformat()
        }
        with open(output_dir / f"{user_id}_start_time.json", 'w') as f:
            json.dump(start_time_data, f)
    
    def generate_complete_user(self, output_dir: Path, user_id: str):
        """Generate a complete user with all 18 files"""
        return self.generate_user_files(output_dir, user_id, num_files=18)
    
    def generate_incomplete_user(self, output_dir: Path, user_id: str, num_files: int = 10):
        """Generate an incomplete user with fewer than 18 files"""
        return self.generate_user_files(output_dir, user_id, num_files=num_files)