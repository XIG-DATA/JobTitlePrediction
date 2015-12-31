var _ = require('lodash');
var fs = require('fs');

var stream = fs.createReadStream('../data/practice.json', {flags: 'r', encoding: 'utf-8'});
var buf = '';

var output_buf = '';
var count = 0;

stream.on('data', function(d) {
    buf += d.toString(); // when data is read, stash it in a string buffer
    pump(); // then process the buffer
});

function pump() {
    var pos;

    while ((pos = buf.indexOf('\n')) >= 0) { // keep going while there's a newline somewhere in the buffer
        if (pos == 0) { // if there's more than one newline in a row, the buffer will now start with a newline
            buf = buf.slice(1); // discard it
            continue; // so that the next iteration will start with data
        }
        processLine(buf.slice(0,pos)); // hand off the line
        count += 1;

        if ( count % 2000 === 0 ) {
            fs.appendFileSync('../data/single.json', output_buf, 'utf8');
            output_buf = '';
            console.log(count);
        }
        buf = buf.slice(pos+1); // and slice the processed data off the buffer
    }
}

function processLine(line) { // here's where we do something with a line

    if (line[line.length-1] == '\r') line=line.substr(0,line.length-1); // discard CR (0x0D)

    if (line.length > 0) { // ignore empty lines
        var results = transform(JSON.parse(line));
        results.forEach(function(r) {
            output_buf += JSON.stringify(r) + '\n';
        });
    }
}

function transform(person) {
    var last_year = person.workExperienceList[0].end_date;
    last_year = last_year ? Number(last_year.split('-')[0]) : 2015;

    var birthday = last_year - Number(person.age); 

    var results = _.cloneDeep(person.workExperienceList);

    var getAge = function(birthday, year) {
        return Number(year) - Number(birthday);
    }

    results.map(function(exp) {
        exp._id = person._id;
        exp.id = _.cloneDeep(person.id);
        exp.degree = person.degree;
        exp.gender = person.gender;
        exp.major = person.major;

        if ( exp.end_date == null ) {
            exp.age = getAge(birthday, last_year);
        } else {
            var current_year = Number(exp.end_date.split('-')[0]);
            exp.age = getAge(birthday, current_year);

            last_year = exp.start_date ? Number(exp.start_date.split('-')[0]) : current_year
        }

    });
    return results;
}