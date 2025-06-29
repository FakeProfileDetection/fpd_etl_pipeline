"""map_to_raw_user_dirs.py"""

import os
import glob
import polars as pl
from pathlib3x import Path
from pprint import pprint

def get_user_file_mapping():
    """
    Create a mapping from user_id to file paths.
    
    returns:
        dict: A dictionary where keys are user_ids and values are lists of file paths.
    """

    # Define mapping from file to user_id, platform_id, session_id, video_id:
    platform_1 = [i for i in range(0, 17, 3)]
    platform_2 = [i + 1 for i in platform_1]
    platform_3 = [i + 2 for i in platform_1]

    # Get session mapping
    session_1 = [i for i in range(0,9)]
    session_2 = [i  for i in range(9,18)]

    # Get video indexes
    video_1 = [0,1,2, 9,10,11]
    video_2 = [i + 3 for i in video_1]
    video_3 = [i + 3 for i in video_2]

    # Get mapping from text (front of filename) to platform index
    platform_idx_to_text = {idx: 'f' for idx in platform_1} 
    platform_idx_to_text.update({idx: 'i' for idx in platform_2})
    platform_idx_to_text.update({idx: 't' for idx in platform_3})

    # Get mapping of file index to platform indexes
    file_index_to_platform = {idx: 1 for idx in platform_1}
    file_index_to_platform.update({idx: 2 for idx in platform_2})
    file_index_to_platform.update({idx: 3 for idx in platform_3})

    # Get mapping of file index to session and video indexes
    file_index_to_session = {idx: 1 for idx in session_1}
    file_index_to_session.update({idx: 2 for idx in session_2})

    file_index_to_video = {idx: 1 for idx in video_1}
    file_index_to_video.update({idx: 2 for idx in video_2})
    file_index_to_video.update({idx: 3 for idx in video_3})

    # Create a map from each file to user_id, platform_id, session_id, video_id
    # where each file index is mapped to its user_id, platform_id, session_id, video_id
    files_mapping = {}
    for u in user_ids:
        for file_index in range(0,18):
            file = f"./uploads/{platform_idx_to_text[file_index]}_{u}_{file_index}.csv"
            # print(f"Processing file: {file}, platform: {file_index_to_platform[file_index]}, session: {file_index_to_session[file_index]}, video: {file_index_to_video[file_index]}")
            files_mapping[file] = {
                "user_id": u,
                "platform_id": file_index_to_platform[file_index],
                "session_id": file_index_to_session[file_index],
                "video_id": file_index_to_video[file_index]
            }
            
    return files_mapping

def move_to_broken_data_dir(new_data_dir, user_id):
    """
    Moves the user directory to the broken_data directory.if the directory already exists, de
    
    :param new_data_dir: Path to the directory where the user directories are stored.
    :param user_id: The user_id of the directory to move.
    
    returns:
    - None
    """
    user_dir = new_data_dir / user_id
    broken_data_dir = new_data_dir / "broken_data"
    broken_data_dir.mkdir(parents=True, exist_ok=True)
    
    if user_dir.exists():
        print(f"Moving {user_dir} to {broken_data_dir}")
        for file in user_dir.iterdir():
            if file.is_file():
                # Move each file to the broken_data directory
                os.system(f"cp {file} {broken_data_dir / file.name}")
                os.system(f"rm {file}")
        # Remove the user directory after moving files
        user_dir.rmdir()
        print(f"Moved {user_dir} to {broken_data_dir}")
    else:
        print(f"User directory {user_dir} does not exist, skipping.")
        pass
        
        
def copy_files_to_user_dirs(old_data_dir, new_data_dir, files_mapping):
    """
    Copies files from old_data_dir to new_data_dir, renaming them according to the mapping.
    
    :param old_data_dir: Path to the directory containing the original files.
    :param new_data_dir: Path to the directory where the renamed files will be saved.
    :param files_mapping: Dictionary mapping file paths to user_id, platform_id, session_id, video_id.
    
    saves files not found and files found to csv files in the new_data_dir.
    
    returns:
    - csv file describing files that are missing or could not be copied
    - csv file describing files that were successfully copied
    """
    
    files_not_found = []
    files_copied = []
    for file, mapping in files_mapping.items():
        # create user_id directory if it doesn't exist
        user_dir = new_data_dir / mapping['user_id']
        user_dir.mkdir(parents=True, exist_ok=True)
        
        new_file_name = f"{mapping['platform_id']}_{mapping['video_id']}_{mapping['session_id']}_{mapping['user_id']}.csv"
        new_file_path = user_dir / new_file_name
        old_path = Path(file) / file
        
        # Copy the file to the new location
        try:
            Path(file).copy(new_file_path)
            print(f"Copied {file} to {new_file_path}")
            files_copied.append({
                "old_file": file,
                "new_file": str(new_file_path),
                "user_id": mapping['user_id'],
                "platform_id": mapping['platform_id'],
                "session_id": mapping['session_id'],
                "video_id": mapping['video_id']
            })
        except FileNotFoundError:
            print(f"File not found: {file}")
            files_not_found.append({
                "file": file,
                "user_id": mapping['user_id'],
                "platform_id": mapping['platform_id'],
                "session_id": mapping['session_id'],
                "video_id": mapping['video_id'],
                "error": "File not found"
            })
            continue
        except Exception as e:
            print(f"Error copying {file} to {new_file_path}: {e}")
            files_not_found.append({
                "file": file,
                "user_id": mapping['user_id'],
                "platform_id": mapping['platform_id'],
                "session_id": mapping['session_id'],
                "video_id": mapping['video_id'],
                "error": str(e)
            })
            continue
        
    # Save the files_not_found and files_copied to CSV files
    if files_not_found:
        not_found_df = pl.DataFrame(files_not_found)
        not_found_df.write_csv(new_data_dir / "files_not_found.csv")
        print(f"Saved {len(files_not_found)} files not found to {new_data_dir / 'files_not_found.csv'}")
    else:
        not_found_df = None
    if files_copied:
        copied_df = pl.DataFrame(files_copied)
        copied_df.write_csv(new_data_dir / "files_copied.csv")
        print(f"Saved {len(files_copied)} files copied to {new_data_dir / 'files_copied.csv'}")
    else:
        copied_df = None
        
    # Test each file in copied_df to see if it can be read into a polars DataFrame
    if copied_df is not None:
        bad_files = []
        bad_user_ids = set(not_found_df['user_id'].to_list()) if not_found_df is not None else set()
        for row in copied_df.iter_rows(named=True):
            user_id = row['user_id']
            file = row['new_file']
            # load into polars dataframe
            if user_id not in bad_user_ids:
                continue
            try:
                df = pl.read_csv(file, has_header=False, infer_schema_length=5000)
                print(f"Loaded {file} with {df.shape[0]} rows and {df.shape[1]} columns.")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                bad_files.append(file)
                continue
            
        # Add bad_files to not_found_df
        if bad_files:
            for file in bad_files:
                user_id = Path(file).parent.name
                platform_id, video_id, session_id, _ = Path(file).stem.split('_')
                not_found_df = not_found_df.append({
                    "file": file,
                    "user_id": user_id,
                    "platform_id": platform_id,
                    "session_id": session_id,
                    "video_id": video_id,
                    "error": "File could not be read into a DataFrame"
                }, ignore_index=True)
        
        
    # Copy data from user ids in not_found_df to "broken_data" directory
    if not_found_df is not None:
        bad_user_ids = not_found_df['user_id'].unique().to_list() if not_found_df is not None else []
        for user_id in bad_user_ids:
            move_to_broken_data_dir(new_data_dir, user_id)
        
        # Remove bad user ids data from copied_df
        if copied_df is not None:
            copied_df = copied_df.filter(~pl.col("user_id").is_in(bad_user_ids))
            if not copied_df.is_empty():
                copied_df.write_csv(new_data_dir / "files_copied.csv")
                print(f"Updated files_copied.csv with remaining files: {new_data_dir / 'files_copied.csv'}")
            else:
                pass
                print("No files left in files_copied after removing bad user ids.")
    else:
        print("No files not found, skipping moving user directories to broken_data.")
        
    if files_not_found:
        not_found_df.write_csv(new_data_dir / "files_not_found.csv")
        print(f"Saved {len(files_not_found)} files not found to {new_data_dir / 'files_not_found.csv'}")

    if files_copied:
        copied_df.write_csv(new_data_dir / "files_copied.csv")
        print(f"Saved {len(files_copied)} files copied to {new_data_dir / 'files_copied.csv'}")
            
    
    return not_found_df, copied_df
if __name__ == "__main__":
    # Define the old and new data directories
    old_data_dir = Path("./uploads")
    new_data_dir = Path("./new_uploads")
    
    # Create the new data directory if it doesn't exist
    new_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the user file mapping
    files_mapping = get_user_file_mapping()
    
    # Copy files to user directories
    not_found_df, copied_df = copy_files_to_user_dirs(old_data_dir, new_data_dir, files_mapping)
    
    # Print the results
    pprint(not_found_df)
    pprint(copied_df)