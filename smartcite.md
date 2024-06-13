# Smartcite jar package construction, startup, and paper parsing
```shell
git checkout git@github.com:pengzh1/SmartCite.git
cd SmartCite
git checkout origin/devzp
mvn clean install
java -jar target/smart_cite-2.0.1-SNAPSHOT.jar
curl http://localhost:8080/localSplit\?path\=/data/path/paper-xml
```
