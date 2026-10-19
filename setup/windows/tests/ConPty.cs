using System;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

public sealed class DevenvConsole : IDisposable {
    [StructLayout(LayoutKind.Sequential)] struct Coord { public short X, Y; }
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)] struct Startup {
        public int cb; public string reserved, desktop, title;
        public int x,y,width,height,xChars,yChars,fill,flags;
        public short show,reservedSize; public IntPtr reservedPtr,input,output,error;
    }
    [StructLayout(LayoutKind.Sequential)] struct StartupEx { public Startup startup; public IntPtr attributes; }
    [StructLayout(LayoutKind.Sequential)] struct ProcessInfo { public IntPtr process,thread; public int id,threadId; }
    [DllImport("kernel32.dll",SetLastError=true)] static extern bool CreatePipe(out IntPtr read,out IntPtr write,IntPtr security,int size);
    [DllImport("kernel32.dll")] static extern int CreatePseudoConsole(Coord size,IntPtr input,IntPtr output,int flags,out IntPtr console);
    [DllImport("kernel32.dll")] static extern void ClosePseudoConsole(IntPtr console);
    [DllImport("kernel32.dll",SetLastError=true)] static extern bool InitializeProcThreadAttributeList(IntPtr list,int count,int flags,ref IntPtr size);
    [DllImport("kernel32.dll",SetLastError=true)] static extern bool UpdateProcThreadAttribute(IntPtr list,int flags,IntPtr attribute,IntPtr value,IntPtr size,IntPtr previous,IntPtr returned);
    [DllImport("kernel32.dll")] static extern void DeleteProcThreadAttributeList(IntPtr list);
    [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern bool CreateProcessW(string app,StringBuilder command,IntPtr processSecurity,IntPtr threadSecurity,bool inherit,int flags,IntPtr environment,string directory,ref StartupEx startup,out ProcessInfo process);
    [DllImport("kernel32.dll",SetLastError=true)] static extern bool WriteFile(IntPtr handle,byte[] data,int count,out int written,IntPtr overlapped);
    [DllImport("kernel32.dll",SetLastError=true)] static extern bool ReadFile(IntPtr handle,byte[] data,int count,out int read,IntPtr overlapped);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll")] static extern uint WaitForSingleObject(IntPtr handle,uint ms);
    [DllImport("kernel32.dll")] static extern bool TerminateProcess(IntPtr handle,uint exitCode);
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern short VkKeyScanW(char ch);
    [DllImport("user32.dll")] static extern uint MapVirtualKeyW(uint code,uint type);
    IntPtr console,input,output,process; Task reader;
    public int ProcessId { get; private set; }
    readonly StringBuilder captured=new StringBuilder();
    void Check(bool success) { if(!success) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error()); }
    public DevenvConsole(string command,string directory) {
        IntPtr childInput,childOutput;
        Check(CreatePipe(out childInput,out input,IntPtr.Zero,0));
        Check(CreatePipe(out output,out childOutput,IntPtr.Zero,0));
        int result=CreatePseudoConsole(new Coord{X=120,Y=40},childInput,childOutput,0,out console);
        CloseHandle(childInput); CloseHandle(childOutput);
        if(result!=0) throw new Exception("CreatePseudoConsole failed: "+result);
        reader=Task.Run(()=>{
            byte[] bytes=new byte[8192]; int read;
            while(ReadFile(output,bytes,bytes.Length,out read,IntPtr.Zero)&&read>0) {
                lock(captured) { captured.Append(Encoding.UTF8.GetString(bytes,0,read)); if(captured.Length>65536) captured.Remove(0,captured.Length-65536); }
            }
        });
        IntPtr size=IntPtr.Zero;
        InitializeProcThreadAttributeList(IntPtr.Zero,1,0,ref size);
        IntPtr attributes=Marshal.AllocHGlobal(size);
        try {
            Check(InitializeProcThreadAttributeList(attributes,1,0,ref size));
            Check(UpdateProcThreadAttribute(attributes,0,(IntPtr)0x20016,console,(IntPtr)IntPtr.Size,IntPtr.Zero,IntPtr.Zero));
            StartupEx start=new StartupEx(); start.startup.cb=Marshal.SizeOf(typeof(StartupEx)); start.attributes=attributes;
            // Clear inherited pipe handles so console I/O goes through this ConPTY.
            start.startup.flags=0x100;
            ProcessInfo info;
            Check(CreateProcessW(null,new StringBuilder(command),IntPtr.Zero,IntPtr.Zero,false,0x80000,IntPtr.Zero,directory,ref start,out info));
            process=info.process; ProcessId=info.id; CloseHandle(info.thread);
        } finally { DeleteProcThreadAttributeList(attributes); Marshal.FreeHGlobal(attributes); }
    }
    public void Send(string text) { byte[] bytes=Encoding.UTF8.GetBytes(text); int written; Check(WriteFile(input,bytes,bytes.Length,out written,IntPtr.Zero)); }
    public void Key(char ch,bool control) {
        if(control) ch=Char.ToLowerInvariant(ch);
        int mapped=VkKeyScanW(ch), vk=mapped&255, state=((mapped>>8)&1)!=0?16:0;
        if(control) state|=8;
        int unicode=control?Char.ToUpperInvariant(ch)-64:ch;
        uint scan=MapVirtualKeyW((uint)vk,0);
        Send("\x1b["+vk+";"+scan+";"+unicode+";1;"+state+";1_");
        Send("\x1b["+vk+";"+scan+";"+unicode+";0;"+state+";1_");
    }
    public string Output { get { lock(captured) return captured.ToString(); } }
    public void Dispose() {
        if(console!=IntPtr.Zero) { ClosePseudoConsole(console); console=IntPtr.Zero; }
        if(process!=IntPtr.Zero) { if(WaitForSingleObject(process,2000)==258) TerminateProcess(process,1); CloseHandle(process); process=IntPtr.Zero; }
        CloseHandle(input); CloseHandle(output);
    }
}
