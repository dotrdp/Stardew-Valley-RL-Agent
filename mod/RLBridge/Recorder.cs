using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Text.Json.Nodes;

namespace RLBridge
{
    /// <summary>
    /// Buffers protocol events (warp, menu_open, day_end, ...) on the game thread. The agent buffer is drained into
    /// each step/macro response; the recorder buffer (filled only while recording) is drained into recording lines.
    /// Events are stored as JSON strings so each consumer gets fresh nodes.
    /// </summary>
    internal sealed class EventLog
    {
        private const int MaxBuffered = 2000;
        private readonly List<string> agent = new();
        private readonly List<string> recorder = new();

        /// <summary>Whether events are also buffered for the recorder.</summary>
        public bool RecordEnabled { get; set; }

        public bool HasRecorderEvents => this.recorder.Count > 0;

        public void Add(string kind, params (string Key, object? Value)[] fields)
        {
            var record = new Dictionary<string, object?> { ["kind"] = kind };
            foreach ((string key, object? value) in fields)
                record[key] = value;
            string json = Json.Serialize(record);

            Push(this.agent, json);
            if (this.RecordEnabled)
                Push(this.recorder, json);
        }

        public JsonArray DrainAgent()
        {
            return Drain(this.agent);
        }

        public JsonArray DrainRecorder()
        {
            return Drain(this.recorder);
        }

        private static void Push(List<string> list, string json)
        {
            if (list.Count >= MaxBuffered)
                list.RemoveAt(0); // nobody is consuming; keep the newest
            list.Add(json);
        }

        private static JsonArray Drain(List<string> list)
        {
            var array = new JsonArray();
            foreach (string json in list)
                array.Add(JsonNode.Parse(json));
            list.Clear();
            return array;
        }
    }

    /// <summary>
    /// Appends gzipped JSONL to a file: a header line, then one line per sampled change:
    /// <c>{"tick", "obs": Snapshot, "input": InputFrame (inferred from held keys), "events": [...]}</c>.
    /// Appending to an existing file adds a new gzip member (multi-member gzip; readers must support it).
    /// </summary>
    internal sealed class Recorder : IDisposable
    {
        private StreamWriter? writer;
        private string? lastObsKey;
        private string? lastInputKey;
        private int linesSinceFlush;

        /// <summary>The current output file, if recording.</summary>
        public string? FilePath { get; private set; }

        public bool Enabled => this.writer != null;

        public long LinesWritten { get; private set; }

        public void Start(string path, JsonObject header)
        {
            this.Stop();
            string? dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);

            var file = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read);
            var gzip = new GZipStream(file, CompressionLevel.Optimal);
            this.writer = new StreamWriter(gzip, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)) { NewLine = "\n" };
            this.FilePath = path;
            this.lastObsKey = null;
            this.lastInputKey = null;
            this.LinesWritten = 0;
            this.WriteLine(header);
            this.writer.Flush();
        }

        /// <summary>Write a line if the snapshot (ignoring its tick), the input or the events changed.</summary>
        public void Sample(ulong tick, JsonObject snapshot, InputFrame input, JsonArray events)
        {
            if (this.writer == null)
                return;

            snapshot.Remove("tick");
            string obsKey = snapshot.ToJsonString();
            JsonObject inputJson = input.ToJson();
            string inputKey = inputJson.ToJsonString();
            if (obsKey == this.lastObsKey && inputKey == this.lastInputKey && events.Count == 0)
                return;
            this.lastObsKey = obsKey;
            this.lastInputKey = inputKey;

            snapshot["tick"] = tick;
            this.WriteLine(new JsonObject
            {
                ["tick"] = tick,
                ["obs"] = snapshot,
                ["input"] = inputJson,
                ["events"] = events,
            });

            if (++this.linesSinceFlush >= 30)
            {
                this.linesSinceFlush = 0;
                this.writer.Flush(); // sync-flushes the gzip stream, so a crash loses at most ~30 lines
            }
        }

        public void Stop()
        {
            if (this.writer == null)
                return;
            try
            {
                this.writer.Dispose(); // writes the gzip trailer and closes the file
            }
            catch (Exception)
            {
                // ignore
            }
            this.writer = null;
            this.FilePath = null;
        }

        public void Dispose()
        {
            this.Stop();
        }

        private void WriteLine(JsonObject line)
        {
            this.writer!.Write(line.ToJsonString());
            this.writer.Write('\n');
            this.LinesWritten++;
        }
    }
}
