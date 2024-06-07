# Docstrings for data collection.

This source code to collect data from youtube video (url)

## Local machine
### Clone repository
```bash
cd dir/contain/project
git clone https://github.com/tanthinhdt/vietnamese-av-asr.git
```
### Install dependencies
```bash
%%bash
cd vietnamese-av-asr
./src/data/scripts/prepare.sh
# If fail, chmod and rerun
```
### Login huggingface-hub
```bash
!huggingface-cli login
# Then add token to the prompt
```
### Process
Track URLs
```bash
python src/data/tasks/track.py --url <url> --channel-name <channel-name>
```
`<url>: url of video in YouTube.
`

`<channel-name>: As alias channel to process.
`

Put channel name to channel.txt file
```python
#channel name forms 'batch_00000'
channel_names = """
batch_99999
"""
with open('src/data/databases/channel.txt','w') as f:
    print(*channel_names, sep='\n',file=f)
```
Process data through each task
```bash
python src/data/tasks/process 
    --task <task> in [download|asd|crop|vndetect|transcribe] 
    --channel-names <channel-names> 
    --cache-dir <cache-dir>  
    --output-dir <output-dir> 
    --upload-to-hub          
    --clean-input            
    --clean-output           
```


- `<task>: Select one task to process.`
- `<channel-names>: Channel name or file contains channel names. Default 'src/data/databases/channel.txt'`
- `<cache-dir>: Directory contains downloaded data from hub. Default 'data/external/'`
- `<output-dir>: Directory contains processed data ready upload to hub. Default 'data/processed/'`
- `--upload-to-hub: Upload processed data to hub.`
- `--clean-input: Clean cache dir.`
- `--clean-output: Clean output dir.`

### Pipeline (demo)
Execute tasks on video file or url.
```bash
python src/data/tasks/pipe.py
    --url <url>                     
    --file <path-to-file>
    --do-file                       
    --channel-name <channel-name>   
    --tasks <task1> <task2> ...     
    --cache-dir <cache-dir>         
    --output-dir <output-dir>       
    --clean-input
    --clean-output
```
- `<url>: Video url in YouTube.`
- `<path-to-file>: Path to video file.`
- `--do-file: Process video file instead of url.`
- `<channel-name>: Name of channel contains url/file.`
- `<task1> <task2>: Select tasks to process, or 'full' to do all tasks.`
- `<cache-dir>: Directory contains downloaded data from hub. Default 'data/external/'`
- `<output-dir>: Directory contains processed data ready upload to hub. Default 'data/processed/'`
- `--clean-input: Clean cache dir.`
- `--clean-output: Clean output dir.`
- `Available tasks in order: track, download, asd, crop, vndetect, transcribe`
- `Notes: Should select tasks consecutively. `

## Colab
### Open browser, go to colab, Open notebook 'data_process.ipynb' in [repo](https://github.com/tanthinhdt/vietnamese-av-asr/tree/main/notebooks).




