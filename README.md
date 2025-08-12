# Dataset Setup Instructions

This project uses the **nuScenes Mini** dataset for evaluating adversarial robustness in autonomous navigation. Follow these instructions to download and set up the dataset.

## Prerequisites

- At least 6GB of free disk space
- Stable internet connection
- `tar` extraction utility (built-in on macOS/Linux, use 7-Zip on Windows)

## Download Instructions

### Step 1: Register for nuScenes Account

1. Visit [https://www.nuscenes.org/nuscenes](https://www.nuscenes.org/nuscenes)
2. Click on "Register" in the top right
3. Fill out the registration form (academic email preferred)
4. Verify your email address

### Step 2: Download nuScenes Mini Dataset

1. Log in to your nuScenes account
2. Navigate to the [Download page](https://www.nuscenes.org/nuscenes#download)
3. Scroll to the **Mini** section
4. Download the following files:
   - `v1.0-mini.tar` (~400 MB) - Metadata and annotations


### Step 3: Extract Dataset

#### macOS/Linux:
```bash
# Navigate to project root
cd path/to/AI_Project_Tekumalla\&Dumpala

# Create data directory if it doesn't exist
mkdir -p data

# Move downloaded files to data directory
mv ~/Downloads/v1.0-mini.tar data/


# Extract files
cd data
tar -xf v1.0-mini.tar


# Clean up (optional)
rm v1.0-mini.tar 
```
#### Windows:

```bash
# Navigate to project root
cd path\to\AI_Project_Tekumalla\&Dumpala

# Create data directory
mkdir data

# Extract using 7-Zip or Windows built-in extraction
# Extract the tar files to the data folder
```