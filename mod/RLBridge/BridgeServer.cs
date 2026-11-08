using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using StardewModdingAPI;

namespace RLBridge
{
    /// <summary>A request line received from the client, answered later from the game thread.</summary>
    internal sealed class PendingRequest
    {
        private readonly ManualResetEventSlim done = new(false);
        private string? response;

        public PendingRequest(string line, bool isDisconnect = false)
        {
            this.Line = line;
            this.IsDisconnect = isDisconnect;
        }

        /// <summary>The raw JSON line.</summary>
        public string Line { get; }

        /// <summary>Whether this is the sentinel queued when the client disconnects.</summary>
        public bool IsDisconnect { get; }

        /// <summary>Send the response (called on the game thread). Later calls are ignored.</summary>
        public void Respond(string json)
        {
            if (this.done.IsSet)
                return;
            this.response = json;
            this.done.Set();
        }

        /// <summary>Wait for the response (called on the network thread). Returns null if cancelled.</summary>
        public string? Wait(CancellationToken token)
        {
            try
            {
                this.done.Wait(token);
                return this.response;
            }
            catch (OperationCanceledException)
            {
                return null;
            }
        }
    }

    /// <summary>
    /// TCP server on a background thread. One client at a time; requests are newline-delimited JSON,
    /// handled strictly in order: the server reads a line, queues it for the game thread, waits for the
    /// answer and writes it back before reading the next line.
    /// </summary>
    internal sealed class BridgeServer : IDisposable
    {
        private readonly string host;
        private readonly int port;
        private readonly IMonitor monitor;
        private readonly CancellationTokenSource cancel = new();
        private TcpListener? listener;
        private Thread? thread;
        private volatile bool connected;

        public BridgeServer(string host, int port, IMonitor monitor)
        {
            this.host = host;
            this.port = port;
            this.monitor = monitor;
        }

        /// <summary>Requests waiting for the game thread.</summary>
        public BlockingCollection<PendingRequest> Queue { get; } = new(new ConcurrentQueue<PendingRequest>());

        /// <summary>Whether a client is connected.</summary>
        public bool IsConnected => this.connected;

        public void Start()
        {
            this.listener = new TcpListener(IPAddress.Parse(this.host), this.port);
            this.listener.Start();
            this.thread = new Thread(this.Run) { IsBackground = true, Name = "RLBridge server" };
            this.thread.Start();
            this.monitor.Log($"Listening on {this.host}:{this.port}.", LogLevel.Info);
        }

        public void Dispose()
        {
            this.cancel.Cancel();
            try
            {
                this.listener?.Stop();
            }
            catch (Exception)
            {
                // ignore
            }
        }

        private void Run()
        {
            while (!this.cancel.IsCancellationRequested)
            {
                TcpClient client;
                try
                {
                    client = this.listener!.AcceptTcpClient();
                }
                catch (Exception ex)
                {
                    if (this.cancel.IsCancellationRequested)
                        return;
                    this.monitor.Log($"Accept failed: {ex.Message}", LogLevel.Warn);
                    Thread.Sleep(250);
                    continue;
                }

                this.Serve(client);
            }
        }

        private void Serve(TcpClient client)
        {
            using (client)
            {
                client.NoDelay = true;
                this.connected = true;
                this.monitor.Log($"Client connected ({client.Client.RemoteEndPoint}).", LogLevel.Info);
                try
                {
                    NetworkStream stream = client.GetStream();
                    var encoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
                    using var reader = new StreamReader(stream, encoding, detectEncodingFromByteOrderMarks: false, bufferSize: 65536, leaveOpen: true);
                    using var writer = new StreamWriter(stream, encoding, bufferSize: 65536, leaveOpen: true);

                    while (!this.cancel.IsCancellationRequested)
                    {
                        string? line = reader.ReadLine();
                        if (line == null)
                            break; // client closed
                        if (string.IsNullOrWhiteSpace(line))
                            continue;

                        var request = new PendingRequest(line);
                        this.Queue.Add(request);
                        string? response = request.Wait(this.cancel.Token);
                        if (response == null)
                            break;

                        writer.Write(response);
                        writer.Write('\n');
                        writer.Flush();
                    }
                }
                catch (IOException)
                {
                    // connection reset
                }
                catch (ObjectDisposedException)
                {
                    // shutting down
                }
                catch (Exception ex)
                {
                    this.monitor.Log($"Connection error: {ex}", LogLevel.Warn);
                }
                finally
                {
                    this.connected = false;
                    if (!this.cancel.IsCancellationRequested)
                        this.Queue.Add(new PendingRequest("", isDisconnect: true)); // wakes a lockstep-blocked game thread
                    this.monitor.Log("Client disconnected.", LogLevel.Info);
                }
            }
        }
    }
}
