# Docstrings for data collection.

This source code to collect data from youtube video (url)

# Local machine
## Clone repository
```bash
git clone -b inference https://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@github.com/tanthinhdt/vietnamese-av-asr.git
# Replace branch 'data_collection' by 'main' if available and necessary.
```
## Prepare environment
Read [here](scripts/README.md) to prepare environment.

## Go to project dir
```bash
cd ./vietnamese-av-asr/
```

## Login huggingface-hub
```bash
huggingface-cli login
# Add token to the prompt
```
### Process
Track URLs
```bash
python src/data/tasks/track.py --url <url> --channel-name <channel-name>
```
- `<url>`: [File contains] url of video in YouTube.
- `<channel-name>`: Name of channel contain url

Put channel name to channel.txt file
```python
#channel name forms 'batch_00000'
channel = """
batch_99999
"""
with open('src/data/databases/channel.txt','w') as f:
    print(*channel, sep='\n',file=f)
```
Process data through each task
```bash
python src/data/tasks/process 
    --task <task>
    --channel-names <channel-names> 
    --cache-dir <cache-dir>  
    --output-dir <output-dir> 
    --upload-to-hub          
    --clean-input            
    --clean-output           
```


- `<task>`: Select one task to process. Task in [download, asd, crop, vndetect, transcribe] 
- `<channel-names>`: Channel name or file contains channel names.
- `<cache-dir>`: Directory contains downloaded data from hub. Default 'data/external/'
- `<output-dir>`: Directory contains processed data ready upload to hub. Default 'data/processed/'
- `--upload-to-hub`: Upload processed data to hub.
- `--clean-input`: Clean cache dir.
- `--clean-output`: Clean output dir.

### Pipeline
Prepare url.
```python
# Paste urls
url = """
https://www.youtube.com/watch?v=xmBdARJx6PY
""".split()
with open("src/data/databases/url.txt", "w") as f:
    print(*url, sep="\n", file=f)
```
Execute tasks on video file or url.
```bash
python src/data/tasks/pipe.py
    --url <url>                     
    --file <path-to-file>    
    --channel-name <channel-name>   
    --tasks <task1> <task2> ...     
    --cache-dir <cache-dir>         
    --output-dir <output-dir>
    --do-file                          
    --clean-input
    --clean-output
    --demo
    --overwrite
```
##### Arguments
- `<url>`: [File contains] url of video in YouTube.
- `<path-to-file>`: Path to video file.
- `<channel-name>`: Name of channel contains url/file.
- `<task1> <task2>`: Select tasks to process, or 'full' to do all tasks.
- `<cache-dir>`: Directory contains downloaded data from hub. Default 'data/external/'
- `<output-dir>`: Directory contains processed data ready upload to hub. Default 'data/processed/'
- `--do-file`: Process video file instead of url.`
- `--clean-input`: Clean cache dir.
- `--clean-output`: Clean output dir.
- `--demo`: Demo
- `--overwrite`: Overwrite

Available tasks in order: track, download, asd, crop, vndetect, transcribe
##### Note: Should select tasks consecutively. 
