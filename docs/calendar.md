**Complete breakdown: OrgBrain Calendar Feature**

---

## **PART 1: What You're Building**

A **Slack-native availability tracker** where:

1. **Employees** post their availability/status
2. **Managers** query who's free/busy without asking
3. **Calendar** shows visual red/green flags by person and date

**User flows:**

```
EMPLOYEE:
/calendar status busy on May 5 from 10am to 12pm for "Client meeting"
→ Saved to DB
→ Confirmation: ✅ Status updated

MANAGER:
/calendar who-is-free May 5
→ Returns list of free people with their timeslots
→ Shows as Slack blocks with avatars

MANAGER:
/calendar show May
→ Returns monthly grid
→ Green (free) / Red (busy/leave) flags
```

---

## **PART 2: Database Structure (MongoDB)**

**Collection name:** `employee_availability`

**Document structure:**
```javascript
{
  _id: ObjectId,
  
  // WHO
  userId: "U123456",              // Slack user ID
  userName: "sarah.sharma",       // Slack username
  userDisplayName: "Sarah Sharma", // Display name
  userEmail: "sarah@company.com",
  teamId: "T123456",              // Which Slack workspace
  
  // WHEN
  dateStart: "2026-05-05",        // YYYY-MM-DD
  dateEnd: "2026-05-05",          // Same day if single day
  timeStart: "10:00",             // HH:MM (24hr)
  timeEnd: "12:00",
  
  // STATUS
  status: "busy",                 // "free", "busy", "leave"
  reason: "Client meeting",       // Optional
  
  // METADATA
  channelId: "C123456",           // Where it was posted
  createdAt: ISODate,
  updatedAt: ISODate,
  timezone: "Asia/Kolkata"
}
```

**Indexes to add:**
```javascript
// Find availability for a specific date
db.employee_availability.createIndex({ dateStart: 1, dateEnd: 1 })

// Find all entries for a user
db.employee_availability.createIndex({ userId: 1, dateStart: 1 })

// Query for who's free on a date
db.employee_availability.createIndex({ status: 1, dateStart: 1 })
```

---

## **PART 3: Slack Commands (Bolt SDK)**

### **Step 1: Parse the `/calendar` command**

User types:
```
/calendar status busy on May 5 from 10am to 12pm for Client meeting
```

You need to extract:
- **action:** `status`
- **status:** `busy`
- **date:** `May 5` → convert to `2026-05-05`
- **timeStart:** `10am` → convert to `10:00`
- **timeEnd:** `12pm` → convert to `12:00`
- **reason:** `Client meeting` (optional)

**Code to parse:**
```javascript
const parseCalendarCommand = (text) => {
  // Input: "status busy on May 5 from 10am to 12pm for Client meeting"
  
  const regex = /^(\w+)\s+(\w+)\s+on\s+(.+?)\s+(?:from\s+(.+?)\s+to\s+(.+?))?(?:\s+for\s+(.+))?$/i;
  const match = text.match(regex);
  
  if (!match) return null;
  
  return {
    action: match[1],        // "status"
    status: match[2],        // "busy"
    dateRaw: match[3],       // "May 5"
    timeStart: match[4],     // "10am"
    timeEnd: match[5],       // "12pm"
    reason: match[6]         // "Client meeting"
  };
};
```

**Convert date formats:**
```javascript
const parseDate = (dateStr, year = 2026) => {
  // "May 5" → "2026-05-05"
  const months = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
  };
  
  const [monthStr, dayStr] = dateStr.toLowerCase().split(/\s+/);
  const month = months[monthStr];
  const day = parseInt(dayStr);
  
  return `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
};

const parseTime = (timeStr) => {
  // "10am" → "10:00", "3:30pm" → "15:30"
  const regex = /^(\d{1,2}):?(\d{2})?\s*(am|pm)?$/i;
  const match = timeStr.match(regex);
  
  if (!match) return null;
  
  let hour = parseInt(match[1]);
  const min = match[2] ? parseInt(match[2]) : 0;
  const period = match[3]?.toLowerCase();
  
  if (period === 'pm' && hour !== 12) hour += 12;
  if (period === 'am' && hour === 12) hour = 0;
  
  return `${String(hour).padStart(2, '0')}:${String(min).padStart(2, '0')}`;
};
```

---

### **Step 2: Slack Command Handler**

```javascript
const { App } = require('@slack/bolt');
const { MongoClient } = require('mongodb');

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  signingSecret: process.env.SLACK_SIGNING_SECRET
});

const mongoClient = new MongoClient(process.env.MONGODB_URI);
const db = mongoClient.db('orgbrain');
const availabilityCollection = db.collection('employee_availability');

app.command('/calendar', async ({ ack, body, client, respond }) => {
  ack(); // Acknowledge immediately (Slack gives 3sec timeout)
  
  const { text, user_id, team_id, channel_id } = body;
  
  const parsed = parseCalendarCommand(text);
  if (!parsed) {
    await respond({
      text: '❌ Invalid format. Use: `/calendar status busy on May 5 from 10am to 12pm for Reason`'
    });
    return;
  }
  
  const { action, status, dateRaw, timeStart, timeEnd, reason } = parsed;
  
  // ACTION: status (post availability)
  if (action === 'status') {
    try {
      const dateStart = parseDate(dateRaw);
      const dateEnd = dateRaw.includes('-') 
        ? parseDate(dateRaw.split('-')[1])  // "May 5-10" → dateEnd = "May 10"
        : dateStart;
      
      const tStart = timeStart ? parseTime(timeStart) : '00:00';
      const tEnd = timeEnd ? parseTime(timeEnd) : '23:59';
      
      // Get user info
      const userInfo = await client.users.info({ user: user_id });
      
      // Save to MongoDB
      const doc = {
        userId: user_id,
        userName: userInfo.user.name,
        userDisplayName: userInfo.user.real_name,
        userEmail: userInfo.user.profile.email,
        teamId: team_id,
        
        dateStart,
        dateEnd,
        timeStart: tStart,
        timeEnd: tEnd,
        
        status,          // "free", "busy", "leave"
        reason: reason || '',
        channelId: channel_id,
        
        createdAt: new Date(),
        updatedAt: new Date(),
        timezone: 'Asia/Kolkata'  // could be user pref
      };
      
      await availabilityCollection.insertOne(doc);
      
      // Respond with confirmation
      await respond({
        blocks: [
          {
            type: 'section',
            text: {
              type: 'mrkdwn',
              text: `✅ *Status Updated*\n📅 ${dateRaw}\n⏰ ${tStart} - ${tEnd}\n🔴 Status: *${status.toUpperCase()}*`
            }
          },
          ...(reason ? [{
            type: 'section',
            text: {
              type: 'mrkdwn',
              text: `📝 Reason: _${reason}_`
            }
          }] : [])
        ]
      });
    } catch (error) {
      await respond({
        text: `❌ Error saving availability: ${error.message}`
      });
    }
  }
  
  // ACTION: who-is-free (query)
  else if (action === 'who-is-free') {
    try {
      const queryDate = parseDate(dateRaw);
      
      // Find all people with any status on this date
      const availabilities = await availabilityCollection
        .find({
          dateStart: { $lte: queryDate },
          dateEnd: { $gte: queryDate }
        })
        .toArray();
      
      // Group by status
      const free = availabilities.filter(a => a.status === 'free');
      const busy = availabilities.filter(a => a.status === 'busy');
      const leave = availabilities.filter(a => a.status === 'leave');
      
      // Build response blocks
      const blocks = [
        {
          type: 'header',
          text: {
            type: 'plain_text',
            text: `📅 ${dateRaw} - Availability`
          }
        },
        {
          type: 'divider'
        },
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `*✅ FREE (${free.length})*`
          }
        },
        ...free.map(f => ({
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `• ${f.userDisplayName}\n  ⏰ ${f.timeStart}-${f.timeEnd}`
          }
        })),
        {
          type: 'divider'
        },
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `*⛔ BUSY (${busy.length})*`
          }
        },
        ...busy.map(b => ({
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `• ${b.userDisplayName}\n  ⏰ ${b.timeStart}-${b.timeEnd}\n  📝 ${b.reason || 'No reason'}`
          }
        })),
        {
          type: 'divider'
        },
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `*🏖️ ON LEAVE (${leave.length})*`
          }
        },
        ...leave.map(l => ({
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `• ${l.userDisplayName}`
          }
        }))
      ];
      
      await respond({
        blocks
      });
    } catch (error) {
      await respond({
        text: `❌ Error querying availability: ${error.message}`
      });
    }
  }
  
  // ACTION: show (monthly calendar)
  else if (action === 'show') {
    try {
      const month = dateRaw; // "May"
      const year = 2026;
      
      // For now, just show a simple view
      // TODO: Build full calendar grid
      
      await respond({
        text: `📅 *${month} ${year} Calendar*\n_Coming soon: Full month view_`
      });
    } catch (error) {
      await respond({
        text: `❌ Error: ${error.message}`
      });
    }
  }
});

// Start the app
(async () => {
  await mongoClient.connect();
  await app.start(process.env.PORT || 3000);
  console.log('⚡️ Bolt app started');
})();
```

---

## **PART 4: Query Logic (Detailed)**

### **Query 1: Find all free people on a specific date**

```javascript
const getFreeOnDate = async (dateStr) => {
  const date = parseDate(dateStr);
  
  // Find ALL entries for that date
  const allEntries = await availabilityCollection.find({
    dateStart: { $lte: date },
    dateEnd: { $gte: date }
  }).toArray();
  
  // Get free people
  const free = allEntries.filter(e => e.status === 'free');
  
  // Get all users in workspace
  const users = await getAllUsers(); // From Slack API
  
  // Find who didn't post anything (assume free)
  const noEntry = users.filter(u => 
    !allEntries.some(e => e.userId === u.id)
  );
  
  return {
    explicit: free,      // Explicitly marked free
    implicit: noEntry    // No entry = free
  };
};
```

### **Query 2: Find common free slot for multiple people**

```javascript
const findCommonSlot = async (dateRange, people, durationMinutes = 60) => {
  // dateRange: ["May 5", "May 10"]
  // people: ["U123", "U456"]
  
  const [startDate, endDate] = dateRange.map(parseDate);
  
  // Find all availability entries for these people in date range
  const entries = await availabilityCollection.find({
    userId: { $in: people },
    dateStart: { $lte: endDate },
    dateEnd: { $gte: startDate }
  }).toArray();
  
  // For each day in range, find gaps
  const slots = [];
  
  for (let d = new Date(startDate); d <= new Date(endDate); d.setDate(d.getDate() + 1)) {
    const dayStr = d.toISOString().split('T')[0];
    
    // Get busy slots for this day
    const busyOnDay = entries.filter(e => 
      e.dateStart <= dayStr && e.dateEnd >= dayStr &&
      e.status !== 'free'
    );
    
    // Check if all people are free on this day
    const allFree = people.every(pid => {
      const busyCount = busyOnDay.filter(b => b.userId === pid).length;
      return busyCount === 0;
    });
    
    if (allFree) {
      slots.push(dayStr);
    }
  }
  
  return slots;
};
```

---

## **PART 5: Edge Cases & Validation**

```javascript
const validateInput = (parsed) => {
  const errors = [];
  
  // Status must be valid
  if (!['free', 'busy', 'leave'].includes(parsed.status)) {
    errors.push('Status must be: free, busy, or leave');
  }
  
  // Date must be in future (or today)
  const today = new Date().toISOString().split('T')[0];
  if (parsed.dateStart < today) {
    errors.push('Cannot set availability for past dates');
  }
  
  // If timeStart and timeEnd provided, validate
  if (parsed.timeStart && parsed.timeEnd) {
    if (parsed.timeStart >= parsed.timeEnd) {
      errors.push('Start time must be before end time');
    }
  }
  
  // Reason should not be too long
  if (parsed.reason && parsed.reason.length > 100) {
    errors.push('Reason too long (max 100 chars)');
  }
  
  return errors;
};
```

---

## **PART 6: Implementation Order (Timeline)**

**Week 1 (Days 1-3): MVP**
- [ ] MongoDB schema + indexes
- [ ] Parse function (date, time, command)
- [ ] `/calendar status` handler → save to DB
- [ ] `/calendar who-is-free` handler → query DB
- [ ] Basic response blocks

**Week 1 (Days 4-5): Testing**
- [ ] Test all date formats ("May 5", "May 5-10", "5/5/26")
- [ ] Test all time formats ("10am", "10:00", "3:30pm")
- [ ] Test with real Slack workspace
- [ ] Handle timezone edge cases

**Week 2: Polish**
- [ ] `/calendar show May` (monthly grid)
- [ ] Slack modal for better UX (optional)
- [ ] Clear/update past entries
- [ ] Better error messages

**Week 3: Marketplace**
- [ ] Marketplace submission
- [ ] Demo video
- [ ] Documentation

---

## **PART 7: File Structure**

```
orgbrain/
├── src/
│   ├── slack_bot.js          # Main Slack app
│   ├── calendar/
│   │   ├── commands.js       # /calendar handler
│   │   ├── parser.js         # Parse date/time
│   │   ├── queries.js        # MongoDB queries
│   │   └── blocks.js         # Slack block responses
│   ├── db.js                 # MongoDB connection
│   └── config.js
├── .env
└── package.json
```

---

**Start here:**

1. **Create MongoDB collection + indexes**
2. **Build parser.js** (date/time parsing)
3. **Build calendar/queries.js** (DB queries)
4. **Build slack_bot.js** (command handler)
5. **Test with real team**

Ready to code? Or want me to explain any step deeper?