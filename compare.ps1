param([Parameter(ValueFromRemainingArguments=$true)]$args)
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\compare.py" @args
