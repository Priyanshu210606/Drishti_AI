param([Parameter(ValueFromRemainingArguments=$true)]$args)
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\train.py" @args
